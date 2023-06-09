from jsputils import paths, nsdorg, nnutils, selectivity
import argparse
import os
from os.path import exists
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import h5py
import time
import copy
from fastprogress import progress_bar
import scipy.stats as stats
from IPython.core.debugger import set_trace
from multiprocessing import Process

from sklearn.linear_model import Lasso, LinearRegression
#from sklearn.random_projection import SparseRandomProjection
#from sklearn.random_projection import johnson_lindenstrauss_min_dim

import torch
import torchvision
import torchlens as tl

##############

parser = argparse.ArgumentParser(description='Fit NSD encoding models')

parser.add_argument('--model-name', default='alexnet-barlow-twins', 
                    type=str, help='model whose features will be fit')

parser.add_argument('--units-for-encoding', default='vpnl-floc-scenes', 
                    type=str, help='units used to encode, either "layer" or <floc-imageset_domain>')

parser.add_argument('--subj', default='subj02', 
                    type=str, help='NSD subject')

parser.add_argument('--space', default='nativesurface', 
                    type=str, help='brain data space (surface or volumetric)')

parser.add_argument('--ROI', default='PPA', 
                    type=str, help='brain data ROI')

parser.add_argument('--ncsnr-threshold', default=0.3, 
                    type=float, help='only encode voxels above a specific noise ceiling')

parser.add_argument('--train-imageset', default='nonshared1000-3rep-batch0', 
                    type=str, help='images used for training the encoding model')

parser.add_argument('--test-imageset', default='nonshared1000-3rep-batch1', 
                    type=str, help='images used for testing the encoding model')

parser.add_argument('--layers-to-analyze', nargs='*', default=[], help='layers to use for model fitting')

parser.add_argument('--overwrite', default=False, 
                    type=bool, help='overwrite existing model fits?')

parser.add_argument('--outer-batch-size', default=250, 
                    type=int, help='how many voxels per chunk of parallelization processes?')

parser.add_argument('--inner-batch-size', default=5, 
                    type=int, help='how many voxels per individual process?')

nsddir = paths.nsd()

def main():
    
    time0 = time.time()
    
    args = parser.parse_args()
        
    layers_to_analyze = nnutils.get_layer_group(args.model_name, args.layers_to_analyze)
    
    savedir = f'{paths.encoding_output_dir()}/{args.subj}-{args.space}-{args.ROI}/{args.model_name}_{args.units_for_encoding}/train-{args.train_imageset}_test-{args.test_imageset}'
    
    os.makedirs(savedir, exist_ok=True)
    print(savedir)
    
    xfm = 'func1pt8_to_anat0pt8_autoFSbbr'
    beta_version = 'betas_fithrf_GLMdenoise_RR'
    betadir = f'{nsddir}/nsddata_betas/ppdata/{args.subj}/{args.space}/{beta_version}'
    
    if args.space == 'nativesurface':
        hemis = ['lh', 'rh']
    elif args.space == 'func1pt8mm':
        raise NotImplementedError('volumetric analysis not implemented (yet)')
        
    # function that takes in a model name and returns 
    # a dictionary with indices of selective units for each domain
    if args.units_for_encoding != 'layer':
        selective_unit_dict = selectivity.get_model_selective_units(args.model_name, args.units_for_encoding, verbose = False)
        
    subj_betas, roi_dfs, include_idx, rep_cocos = nsdorg.load_betas(args.subj, 
                                                                    args.space, 
                                                                    voxel_group = args.ROI,
                                                                    ncsnr_threshold = args.ncsnr_threshold,
                                                                    plot=True)
    
    image_data, brain_data = nsdorg.get_NSD_train_test_images_and_betas(subj = args.subj,
                                                 space = args.space,
                                                 subj_betas = subj_betas,
                                                 rep_cocos = rep_cocos,
                                                 train_imageset = args.train_imageset,
                                                 test_imageset = args.test_imageset)
    
    
    activations = nnutils.get_NSD_train_test_activations(args.model_name, image_data)
    
    ################################
    
    for layer in progress_bar(layers_to_analyze):
    
        layer_savedir = f'{savedir}/{layer}'
        os.makedirs(layer_savedir, exist_ok=True)

        print(layer_savedir)

        X_train = copy.deepcopy(activations['train'][layer])
        X_test = copy.deepcopy(activations['test'][layer])

        # flatten from 4D to 2D if necessary
        dims = X_train.shape

        if len(dims) > 2:
            X_train = np.reshape(X_train, (dims[0], np.prod(dims[1:])))
            dims = X_test.shape
            X_test = np.reshape(X_test, (dims[0], np.prod(dims[1:])))
            
        # if encoding selective units, isolate them
        if args.units_for_encoding != 'layer':
            X_train = X_train[:, selective_unit_dict[layer]['selective_idx']]
            X_test = X_test[:,   selective_unit_dict[layer]['selective_idx']]
            
        # skip if there are 0 features
        if X_train.shape[1] > 0:

            for h, hemi in enumerate(hemis):

                nvox = brain_data['train'][hemi].shape[1]

                if nvox > 0:

                    voxel_labels = list(roi_dfs[h][include_idx[hemi]].index)
                    voxel_labels = [f"{vlab.split(',')[0][1:]}_{vlab.split(',')[1][1:-1]}" for vlab in voxel_labels]
                    voxel_savefns = np.array([f'{layer_savedir}/{vlab}.npy' for vlab in voxel_labels])

                    obatch_start_idx = list(range(0, nvox, args.outer_batch_size))
                    n_obatches = len(obatch_start_idx)

                    for obatch_num, this_obatch_start in enumerate(obatch_start_idx):

                        this_obatch_end = this_obatch_start + args.outer_batch_size

                        y_train = copy.deepcopy(brain_data['train'][hemi][:, this_obatch_start : this_obatch_end])
                        y_test = copy.deepcopy(brain_data['test'][hemi][:, this_obatch_start : this_obatch_end])

                        print(X_train.shape, y_train.shape,
                              X_test.shape, y_test.shape)

                        this_voxel_savefns = voxel_savefns[this_obatch_start : this_obatch_end]

                        print(f'fitting {hemi} outer batch {obatch_num+1} of {n_obatches}')

                        procs = []

                        if this_obatch_end > nvox:
                            idx_required = nvox - this_obatch_start
                            ibatch_start_idx = list(range(0, idx_required, args.inner_batch_size))  
                        else:
                            ibatch_start_idx = list(range(0, args.outer_batch_size, args.inner_batch_size))
                        n_ibatches = len(ibatch_start_idx)

                        # instantiating process with arguments
                        for ibatch_num, this_ibatch_start in enumerate(progress_bar(ibatch_start_idx)):

                            this_ibatch_end = this_ibatch_start + args.inner_batch_size

                            absolute_start_idx = this_obatch_start+this_ibatch_start
                            absolute_end_idx = this_obatch_start+this_ibatch_end

                            print(f'\t\tfitting inner group {ibatch_num+1} of {n_ibatches}. voxel indices {absolute_start_idx} to {absolute_end_idx}')

                            encoding_inputs = {'X_train': copy.deepcopy(X_train),
                                             'X_test': copy.deepcopy(X_test),
                                             'y_train': copy.deepcopy(y_train[:, this_ibatch_start : this_ibatch_end]),
                                             'y_test':  copy.deepcopy(y_test[:,  this_ibatch_start : this_ibatch_end]),
                                             'voxel_savefns': copy.deepcopy(this_voxel_savefns[this_ibatch_start : this_ibatch_end]),
                                             'overwrite': args.overwrite
                                             }

                            proc = Process(target=fit_save_encoding_model, 
                                           args=(encoding_inputs,))
                            procs.append(proc)
                            proc.start()

                        # complete the processes
                        for proc in procs:
                            proc.join() 
                else:

                    print(f'no voxels in ROI {hemi}.{args.ROI} for subj {args.subj}. skipping.')
        else:
            
            print(f'no features in {args.model_name} layer {layer} for unit group {args.units_for_encoding}. skipping.')
    
    time1 = time.time()
    
    print(f'time elapsed: {time1 - time0} sec.')
    print('...done.')
    
    return


def fit_save_encoding_model(encoding_inputs):
    
    fit_model = False
    
    if not encoding_inputs['overwrite']: 
        for voxel_savefn in encoding_inputs['voxel_savefns']:
            if exists(voxel_savefn):
                continue
            else:
                fit_model = True
                #print(voxel_savefn)
                #print('overwrite is False and missing voxel encoding model(s) detected. fitting encoding model')
                break
    else:
        fit_model = True
        
    if fit_model:
        #print('\t\t\trunning encoding model')
        
        mdl = Lasso(positive=True, alpha = 0.1, random_state = 365, 
                    selection = 'random', tol = 1e-3, fit_intercept=True)
        
        mdl.fit(encoding_inputs['X_train'], 
                encoding_inputs['y_train'])

        y_pred = mdl.predict(encoding_inputs['X_test'])

        for v, voxel_savefn in enumerate(encoding_inputs['voxel_savefns']):

            out = dict()
            out['coef'] = mdl.sparse_coef_[v]
            out['intercept'] = mdl.intercept_[v]
            if np.ndim(y_pred) == 1:
                out['y_pred'] = y_pred
            else:
                out['y_pred'] = y_pred[:,v]
            out['r'] = stats.pearsonr(encoding_inputs['y_test'][:,v], out['y_pred'])[0]

            np.save(voxel_savefn, out, allow_pickle=True)
    else:
        print('\t\t\tskipping encoding model, all files exist and overwrite is False.')
        
if __name__ == '__main__':
    main()
        
