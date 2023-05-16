import argparse
from os.path import exists
import torch
import torchvision
import gc
from torchvision import datasets
import torchlens as tl
import numpy as np
import copy
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt
from fastprogress import progress_bar
from IPython.core.debugger import set_trace

from jsputils import paths, statsmodels, nnutils, plotting


parser = argparse.ArgumentParser(description='Identify DNN domain-selective units')

parser.add_argument('--model-name', default='alexnet-supervised', 
                    type=str, help='model whose features will be analyzed')

parser.add_argument('--selective-units', default='vpnl-floc-faces', 
                    type=str, help='localizer imageset to use + domain to extract')

parser.add_argument('--FDR-p', default=0.05, 
                    type=float, help='FDR correction alpha value')

parser.add_argument('--overwrite', default=False, 
                    type=bool, help='overwrite selectivity dicts?')

parser.add_argument('--verbose', default=True, 
                    type=bool, help='show print statements and plots?')


def main():
    
    args = parser.parse_args()
    
    selective_unit_dict = get_model_selective_units(args.model_name, 
                                                    args.selective_units,
                                                    args.FDR_p,
                                                    args.overwrite,
                                                    args.verbose)
    
    return

def run_dnn_localizer_procedure(DNN, fLOC, FDR_p, overwrite, verbose): 
    
    if verbose:
        print('getting DNN selective units...\n')
    
    all_selective_units = dict()
    
    activations_computed = False
    
    #if not hasattr(DNN, 'floc_features'):
    #    activations_computed = False
    #else:
    #    activations_computed = True
        
    for target_domain in fLOC.floc_domains:
    
        savefn = f"{paths.selective_unit_dir()}/{DNN.model_name}_{fLOC.image_set_name}-{target_domain}_FDR-{str(FDR_p)[2:]}.npy"

        if verbose:
            print(savefn, '\n', target_domain)
    
        if exists(savefn) and overwrite is False:
    
            all_selective_units[target_domain] = np.load(savefn,allow_pickle=True).item()
    
        else:

            target_domain_val = np.squeeze(np.argwhere(fLOC.floc_domains == target_domain))

            ## visualize, for sanity
            if verbose:

                print(fLOC.floc_domains)
                print(fLOC.floc_domains[target_domain_val], target_domain_val)
                print('# localizer images in target domain:')
                print(np.sum(fLOC.img_domain_indices == target_domain_val))
                
            if not activations_computed:
                
                print('computing floc set activations')
                
                DNN.get_floc_features(fLOC, field = 'floc_features', device = 'cuda:0')

                activations_computed = True
                
            else: 
                
                print('skipping activation computation - already stored in memory')

            ################

            selective_units = dict()

            for layer in progress_bar(DNN.layer_names_fmt):

                Y = DNN.floc_features[layer]

                if Y.ndim > 2:
                    Y = Y.reshape(Y.shape[0],Y.shape[1]*Y.shape[2]*Y.shape[3])

                if verbose:
                    print(Y.shape)

                selective_units[layer] = compute_selectivity(Y, 
                                                             fLOC.img_domain_indices, 
                                                             target_domain_val,
                                                             FDR_p,
                                                             verbose)

            np.save(savefn, selective_units, allow_pickle=True)
                
            all_selective_units[target_domain] = selective_units
            
    all_selective_units['floc_imageset'] = fLOC.image_set_name
    all_selective_units['floc_domains'] = fLOC.floc_domains
    all_selective_units['model_name'] = DNN.model_name
    all_selective_units['FDR_p'] = FDR_p

    if verbose:
        plotting.plot_selective_unit_props(all_selective_units)
    
        print('\n...done.')
    
    return all_selective_units
    


def compute_selectivity(Y, all_domain_idx, target_domain_val, FDR_p, verbose=True):
    
    assert(Y.ndim == 2)
    n_neurons_in_layer = Y.shape[1]
    
    unique_domain_vals = np.unique(all_domain_idx)
    
    target_domain_idx = all_domain_idx == target_domain_val
    
    # get data from curr domain
    Y_curr = copy.deepcopy(Y[target_domain_idx])
    
    if verbose:
        print(f'\t{np.sum(target_domain_idx)} of {len(all_domain_idx)} images are from the target domain ({target_domain_val})')
        print(f'\tsize of layer is {n_neurons_in_layer} units.')
        print(f'\tshape of Y_curr is {Y_curr.shape}')
    
    pairwise_selective_idx = []
    pairwise_tvals = []
    pairwise_pvals = []
    
    for this_domain_val in unique_domain_vals:
        
        # skip if test was domain vs. same domain
        if this_domain_val != target_domain_val:
        
            Y_test = copy.deepcopy(Y[all_domain_idx==this_domain_val])
            
            # calculate t and p maps
            t,p = stats.ttest_ind(Y_curr, Y_test, axis=0)
            
            # determine which neurons remain significant after FDR correction
            # https://stats.stackexchange.com/questions/63441/what-are-the-practical-differences-between-the-benjamini-hochberg-1995-and-t
            reject, pvals_corrected, _, _ = statsmodels.multipletests(p, alpha=FDR_p, method='FDR_by', 
                                is_sorted=False, returnsorted=False)
            
            # dom neurons have t > 0 and reject is True
            pairwise_selective_idx.append(np.logical_and(reject == True, t > 0))
            pairwise_tvals.append(t)
            pairwise_pvals.append(pvals_corrected)
            
    out = dict()
    out['mask'] = np.all(np.vstack(pairwise_selective_idx), axis = 0)
    out['tval'] = np.nanmean(np.vstack(pairwise_tvals), axis = 0)
    out['pval'] = np.nanmean(np.vstack(pairwise_pvals), axis = 0)
    
    if verbose:
        print(f"\t\tfinal size of region for {target_domain_val} is {np.sum(out['mask'])} units.")
        
    return out

if __name__ == '__main__':
    main()
     