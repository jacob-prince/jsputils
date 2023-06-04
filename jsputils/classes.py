import torch
from os.path import exists
from torch import nn
from torch.utils.data import Dataset
import torchvision
import copy
import PIL.Image as Image
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
from jsputils import paths, nsdorg, plotting, encoding, nnutils, selectivity, validation, lesioning, feature_extractor, readout
from fastprogress import progress_bar
import gc
from IPython.core.debugger import set_trace
from sklearn.linear_model import Lasso, LinearRegression
import scipy.stats as stats
from scipy.spatial.distance import pdist

class NSDImageSet(Dataset):
    def __init__(self, numpy_images, transforms=None):
        self.data = numpy_images
        self.transforms = transforms

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image = self.data[idx]
        #image = torch.from_numpy(image)

        if self.transforms:
            image = self.transforms(image)

        return image

       
class ImageSet:
    def __init__(self, image_set_name, transforms = None):
        
        self.image_set_name = image_set_name
        self.image_folder = f'{paths.image_set_dir()}/{self.image_set_name}'
        self.transforms = transforms
        
        if image_set_name == 'vpnl-floc':
            self.categ_nimg = 144
            self.floc_subdirs = np.array(['adult','body','car','child','corridor','house',
                                                       'instrument','limb','number','scrambled','word'])
            self.floc_domains = np.array(['faces','bodies','objects','scenes','characters','scrambled'])
            self.domain_colors = np.array(['tomato','dodgerblue','orange','limegreen','purple','darkgray'])
            self.subdir_domain_ref = np.array([0,1,2,0,3,3,2,1,4,5,4])
            
        elif image_set_name == 'classic-categ':
            self.categ_nimg = 80
            self.floc_subdirs = np.array(['1-Faces','2-Bodies','3-Scenes','4-Words',
                                          '5-Objects','6-ScrambledObjects','7-ScrambledWords'])
            self.floc_domains = np.array(['faces','bodies','scenes','characters','objects','scrambled-objects','scrambled-words'])
            self.domain_colors = np.array(['tomato','dodgerblue','limegreen','purple','orange','darkgray'])
            self.subdir_domain_ref = np.array([0,1,2,3,4,5,6])
        
        elif image_set_name == 'paired-objects':
            
            self.categ_nimg = 156
            self.floc_subdirs = np.array(['images'])
            self.subdir_domain_ref = np.array([0])
            
        for subdir in os.listdir(self.image_folder):
            if os.path.isdir(subdir):
                assert(len(os.listdir(f'{self.image_folder}/{subdir}')) == self.categ_nimg)
            
        self.img_domain_indices = np.repeat(self.subdir_domain_ref, self.categ_nimg)
        
        self.load_images()
            
    def load_images(self):
        # Load images from the folder using torchvision and apply transforms
        self.images = torchvision.datasets.ImageFolder(self.image_folder, transform = self.transforms)
    
    def plot_domain_labels(self):
        plt.figure()
        plt.plot(self.img_domain_indices)
        plt.yticks(np.arange(len(self.floc_domains)),self.floc_domains);
        plt.xlabel('floc img indices')
        plt.title(f'domain of each floc image: {self.image_set_name}');
        
#############

class DataLoaderFFCV:
    def __init__(self, partition, device = 'cuda:0', batch_size = 512, num_workers = 64, resolution = 224, normalize = True):
        self.partition = partition
        self.device = device
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.resolution = resolution
        self.normalize = normalize
        if self.partition == 'train':
            pass
        elif self.partition == 'val':
            self.data_loader = validation.create_val_loader(paths.ffcv_imagenet1k_valset(), 
                                                            self.device,
                                                            self.num_workers, 
                                                            self.batch_size,
                                                            self.resolution,
                                                            self.normalize)
            
# class DataLoaderTorch:
#     def __init__(self, dataset, batch_size = 512, num_workers = 4, shuffle = False, pin_memory = False):
        
#         self.dataset = dataset
#         self.batch_size = batch_size
#         self.num_workers = num_workers
#         self.shuffle = shuffle
#         self.pin_memory = pin_memory
        
#         self.data_loader = torch.utils.data.DataLoader(
#             dataset=self.dataset,
#             batch_size=self.batch_size,
#             num_workers=self.num_workers,
#             shuffle=self.shuffle,
#             pin_memory=self.pin_memory,
#         )

        

class DNNModel:
    def __init__(self, model_name):
                         
        self.model, self.transforms, self.state_dict, self.is_categ_supervised, self.model_name = nnutils.load_model(model_name)
        
        self.layer_names_fmt, self.layer_names_torch = feature_extractor.get_pretty_layer_names(self.model)
                 
        
    def get_floc_features(self, ImageSet, field = 'floc_features', device = 'cuda:0', invert = False):
        
        data_loader = torch.utils.data.DataLoader(
                    dataset=ImageSet.images,
                    batch_size=len(ImageSet.images),
                    num_workers=1,
                    shuffle=False,
                    pin_memory=False
                )

        image_tensors, _ = next(iter(data_loader))
        
        if invert:
            image_tensors = torch.flip(image_tensors, dims=[2])
        
        setattr(self, field, feature_extractor.get_features(copy.deepcopy(self.model), 
                        image_tensors, self.layer_names_torch, self.layer_names_fmt, device = device))

        del image_tensors, data_loader
        torch.cuda.empty_cache()
        gc.collect()
        
    def get_nsd_features(self, image_matrix, field, device = 'cuda:0'):
        
        transforms_list = [
            # Convert numpy array to PIL Image
            torchvision.transforms.Lambda(lambda x: Image.fromarray(x.numpy()))
        ] 

        # Add the original transforms to the list
        transforms_list.extend(self.transforms.transforms)

        # Create a new Compose object with the updated list
        transforms_list = torchvision.transforms.Compose(transforms_list)
        
        nsd_dataset = NSDImageSet(image_matrix, transforms = transforms_list)
        
        data_loader = torch.utils.data.DataLoader(
                    nsd_dataset,
                    batch_size=len(nsd_dataset),
                    num_workers=1,
                    shuffle=False,
                    pin_memory=False
                )
        
        image_tensors = next(iter(data_loader))
        
        setattr(self, field, feature_extractor.get_features(copy.deepcopy(self.model), 
                        image_tensors, self.layer_names_torch, self.layer_names_fmt, device = device))
        
        del image_tensors, data_loader
        torch.cuda.empty_cache()
        gc.collect()

    def find_selective_units(self, localizer_image_set, FDR_p = 0.05, overwrite = False, verbose = False):
        
        # Define the indices of category-selective units using localizer images
        floc_imageset = ImageSet(localizer_image_set, transforms = self.transforms)
        
        self.selective_units = selectivity.run_dnn_localizer_procedure(self, floc_imageset, FDR_p, overwrite, verbose)
        
    def get_imagenet_accs(self, topk = 5): 
        
        ValLoader = DataLoaderFFCV('val')
        
        self.imagenet_accs = validation.get_imagenet_class_accuracies(copy.deepcopy(self.model).half().to(ValLoader.device),
                                                                      ValLoader.data_loader,
                                                                      ValLoader.device, topk = topk)
        
        del ValLoader
        gc.collect()
        torch.cuda.empty_cache()
        
    def append_readout_layer(self, readout_from):
        self.readout_from = readout_from
        
        readout.append_readout_layer(self, readout_from)
            
    # todo add more training args as inputs
    def train_linear_probe(self, readout_from):
        
        self.readout_from = readout_from
        
        # Train a linear ImageNet probe for self-supervised models
        readout.train_readout_layer(self, readout_from)
        
    def load_readout_weights(self, description, device):
        
        ckpt_fn = f'{paths.training_checkpoint_dir()}/{description}/checkpoint.pth'
        
        ckpt = torch.load(ckpt_fn,
                          map_location=device)

        best_acc = ckpt['best_acc']
        self.readout_model.load_state_dict(ckpt['model'])
        print(best_acc)
        
        
    def assess_lesioning_impact(self):
        # Assess the impact of lesioning selective units on the model's ImageNet performance
        pass


class LesionModel(DNNModel):
    def __init__(self, DNNModel, device):
        
        self.model = DNNModel.model
        if hasattr(DNNModel,'selective_units'):
            self.selective_units = DNNModel.selective_units
        self.device = device
        
        # initialize lesioning model using modules from source network
        masks = dict()
        masks['apply'] = False
        self.model = lesioning.LesionNet(self.model, masks).eval().to(self.device)
        
        self.layer_dims = lesioning.get_layer_dims(self.model, self.device)
    
        
    def apply_randomized_lesions(self, lsn_layer, p = 0.5):
        
        self.model.masks = lesioning.get_random_lesioning_masks(self.model, 
                                                          self.layer_dims, 
                                                          lsn_layer, 
                                                          self.device, 
                                                          p)
        
    def apply_channelized_lesions(self, domain, method):
        self.lsn_method = method
        self.model.masks = lesioning.get_channelized_lesioning_masks(self,
                                                                     self.layer_dims,
                                                                     domain,
                                                                     method,
                                                                     self.device)
        
    def remove_randomized_lesions(self):
        masks = dict()
        masks['apply'] = False
        self.model.masks = masks
                
    def get_imagenet_accs(self, topk = 5): 
        
        ValLoader = DataLoaderFFCV('val')
        
        self.imagenet_accs = validation.get_imagenet_class_accuracies(copy.deepcopy(self.model).half().to(ValLoader.device),
                                                                      ValLoader.data_loader,
                                                                      ValLoader.device, topk = topk)
    
    def get_selective_unit_acts(self, layers):
        
        assert(hasattr(self, 'selective_units'))
        
        if self.model.return_acts is False:
            self.model.return_acts = True
            
        ValLoader = DataLoaderFFCV('val')
        
        print(layers)
        
        self.selective_unit_acts = validation.get_selective_unit_acts(copy.deepcopy(self.model).half().to(ValLoader.device),
                                                                      copy.deepcopy(self.selective_units),
                                                                      list(layers),
                                                                      ValLoader.data_loader,
                                                                      ValLoader.device)
        del ValLoader
        gc.collect()
        torch.cuda.empty_cache()

class fMRISubject:
    def __init__(self, subj, space, beta_version):
        self.subj = subj
        self.space = space
        self.beta_version = beta_version
        self.betadir = f'{paths.nsd()}/nsddata_betas/ppdata/{self.subj}/{self.space}/{self.beta_version}'   
        self.processed_datadir = f'{paths.brain_region_savedir()}/{self.subj}'
        os.makedirs(self.processed_datadir,exist_ok=True)

        if self.space == 'nativesurface':
            self.hemis = ['lh', 'rh']
        elif self.space == 'func1pt8mm':
            self.hemis = ['full']
        
        if not hasattr(self, 'metadata'):
            self.get_voxel_metadata()
    
    def get_voxel_metadata(self):
        self.metadata = nsdorg.load_voxel_info(self.subj, self.space, self.beta_version)
        
    def plot_data(self, data, title):
        plotting.plot_ROI_flatmap(self.subj, self.space, title, data, vmin = np.nanmin(data),
                                  vmax=np.nanmax(data),colorbar=True)
        
            
    def load_fmri_data(self):
        # Load fMRI data from the given path and preprocess it
        pass

    def find_brain_regions(self):
        # Find brain regions with selectivities and add them to self.brain_regions
        pass
    
class BrainRegion(fMRISubject):
    def __init__(self, subj, roi, ncsnr_threshold = 0, plot = False):
        
        self.savefn = f'{subj.processed_datadir}/{subj.space}_{roi}.npy'
        
        if exists(self.savefn):
            print('betas already saved. loading...')
            self.betas = np.load(self.savefn,allow_pickle=True).item().betas
            self.rep_cocos = self.betas['rep_cocos']
            self.included_voxel_idx = self.betas['included_voxel_idx']
        
        if hasattr(subj,'metadata'):
            self.metadata = subj.metadata
        super().__init__(subj.subj, subj.space, subj.beta_version)
        self.roi = roi
        self.ncsnr_threshold = ncsnr_threshold
        self.ncsnr_mask = None

        self.roi_indices = nsdorg.get_voxel_group(self.subj, self.space, self.roi,
                                                  self.ncsnr_threshold, self.metadata, plot = plot)
        self.get_ncsnr()

    def get_ncsnr(self):
        
        self.ncsnr = []
        if self.space == 'nativesurface':
            self.ncsnr = np.concatenate((self.metadata[0]['lh.ncsnr'].values,
                                         self.metadata[1]['rh.ncsnr'].values))

            self.ncsnr[np.logical_not(self.roi_indices['full'])] = np.nan
        elif self.space == 'func1pt8mm':
            raise ValueError('not implemented yet')
        
    def load_betas(self):
        
        if not hasattr(self, 'betas'):
            self.betas, _, self.included_voxel_idx, self.rep_cocos = nsdorg.load_betas(self.subj, 
                                                                                 self.space, 
                                                                                voxel_group = self.roi,
                                                                                ncsnr_threshold = self.ncsnr_threshold,
                                                                                plot=False
                                                                                )
            
            self.betas['rep_cocos'] = self.rep_cocos
            self.betas['included_voxel_idx'] = self.included_voxel_idx
            self.save()
            
    def load_encoding_data(self, train_imageset, val_imageset, test_imageset):
        print('loading train, val, and test data')
        self.image_data, self.brain_data = nsdorg.get_NSD_encoding_images_and_betas(subj = self.subj,
                                                                                      space = self.space,
                                                                                      subj_betas = self.betas,
                                                                                      rep_cocos = self.rep_cocos,
                                                                                      train_imageset = train_imageset,
                                                                                      val_imageset = val_imageset,
                                                                                      test_imageset = test_imageset,
                                                                                      mean = False)
        
    def get_ncsnr_mask(self, threshold):
        self.ncsnr_threshold_ = threshold
        self.ncsnr_mask = self.ncsnr[self.included_voxel_idx['full']] > self.ncsnr_threshold_

    def save(self):
        np.save(self.savefn, self, allow_pickle=True)


class EncodingProcedure:
    def __init__(self, ROI, DNN, method, positive, alphas):
        self.ROI = ROI
        self.DNN = DNN
        self.method = method
        self.positive = positive
        self.alphas = alphas
        self.layers_to_encode = self.DNN.layer_names_fmt
        
    def init_mapper(self, method, positive, alpha):
        self.mdl = get_mapper(method, positive, alpha)
        
    def get_encoding_features(self, layer, domain):
        
        if domain != 'layer':
            mask = self.DNN.selective_units[domain][layer]['mask']
        
        X = {'train': copy.deepcopy(self.DNN.nsd_train_features[layer]),
             'val':   copy.deepcopy(self.DNN.nsd_val_features[layer]),
             'test':  copy.deepcopy(self.DNN.nsd_test_features[layer])}
        
        for partition in X:
            X[partition] = np.reshape(X[partition], (X[partition].shape[0], -1))
            if domain != 'layer':
                X[partition] = X[partition][:, mask]
            print('\t\t', partition, X[partition].shape)
            
        return X
    
    def get_encoding_voxels(self):
            
        y = dict()
        
        for partition in ['train','val','test']:
            y[partition] = np.concatenate((self.ROI.brain_data[partition]['lh'],
                                           self.ROI.brain_data[partition]['rh']), axis = 2)
            y[partition] = np.mean(y[partition],axis = 1)
            
            print('\t\t', partition, y[partition].shape)
            
            if self.ROI.ncsnr_mask is not None:
                print('\t\tmasking')
                y[partition] = y[partition][..., self.ROI.ncsnr_mask]
                print('\t\t',partition, y[partition].shape)
                
        return y
        
    def encode_layers(self, savedir, layers=None, domains = None, overwrite = False):
            
        if layers is None:
            layers = self.layers_to_encode
            
        if domains is None:
            domains = self.DNN.selective_units['floc_domains']
            
        for layer in layers:
            
            for domain in domains:
                
                self.results_df = pd.DataFrame(columns=['subj','ROI','ncsnr_threshold','model_name','layer', 'domain', 'method', 'positive', 'alpha', 'partition', 'cUnivar', 'cRSA', 'veUnivar', 'veRSA'])
       
                model_data_loaded = False
                brain_data_loaded = False
                
                print(self.ROI.subj, self.ROI.roi, layer, domain)
                
                savefn = f'{savedir}/{self.ROI.subj}_{self.ROI.roi}_nc-{self.ROI.ncsnr_threshold_}_{self.DNN.model_name}_{layer}_{domain}.csv'
                                
                # check if output exists
                if exists(savefn) and not overwrite:
                    print('\tresults already computed. loading...')
                    self.results_df = pd.concat([self.results_df, pd.read_csv(savefn)], ignore_index=True)
                    
                else:
                    
                    if not model_data_loaded: 
                        print('\tgetting NSD image features...')
                        for partition in progress_bar(['train','val','test']):
                            self.DNN.get_nsd_features(torch.from_numpy(self.ROI.image_data[partition]), field = f'nsd_{partition}_features')
                            torch.cuda.empty_cache()
                            gc.collect()
                            model_data_loaded = True
                            
                    if not brain_data_loaded:
                        print('\tpreparing encoding brain data...')
                        y = self.get_encoding_voxels()

                        trues = {'val': {},
                                 'test': {}}
                        for pt in trues:
                            trues[pt]['univar'] = np.mean(y[pt],axis=1)
                            trues[pt]['rsa'] = pdist(y[pt],'correlation')
                            
                        brain_data_loaded = True
                    
                    print(f'\tpreparing encoding model features... {layer} {domain}')
                    X = self.get_encoding_features(layer, domain)

                    for alpha in self.alphas:

                        self.init_mapper(self.method, self.positive, alpha)

                        print(f'\tfitting model...',self.mdl)

                        if X['train'].shape[1] > 1:

                            self.mdl.fit(X['train'], y['train'])

                            print(f'\tgenerating predictions...')

                            preds = {'val': {},
                                     'test': {}}
                            for pt in preds:
                                pred_y = self.mdl.predict(X[pt])
                                preds[pt]['univar'] = np.mean(pred_y,axis=1)
                                preds[pt]['rsa'] = pdist(pred_y,'correlation')

                            print(f'\tcomputing metrics...')

                            # cUnivar
                            # cRSA
                            # veUnivar
                            # veRSA
                            for pt in ['val','test']:

                               # try:
                                cUnivar = stats.pearsonr(np.nanmean(X[pt],axis=1),
                                                         trues[pt]['univar'])[0]

                                model_crdv = pdist(X[pt],'correlation')

                                prop_bad, valid = check_nan_or_inf(model_crdv)

                                if prop_bad > 0.1:
                                    print(prop_bad)
                                    cRSA = np.nan
                                else:
                                    cRSA = stats.pearsonr(model_crdv[valid],
                                                          trues[pt]['rsa'][valid])[0]

                                veUnivar = stats.pearsonr(preds[pt]['univar'],
                                                          trues[pt]['univar'])[0]

                                model_verdv = preds[pt]['rsa']
                                prop_bad, valid = check_nan_or_inf(model_verdv)

                                if prop_bad > 0.1:
                                    print(prop_bad)
                                    veRSA = np.nan
                                else:
                                    veRSA = stats.pearsonr(preds[pt]['rsa'][valid],
                                                           trues[pt]['rsa'][valid])[0]

                                print('\t\t\t',pt, 'cUnivar',round(cUnivar,3), 'cRSA',round(cRSA,3), 'veUnivar',round(veUnivar,3), 'veRSA',round(veRSA,3))

                                result_row = pd.DataFrame({'subj': [self.ROI.subj], 
                                                           'ROI': [self.ROI.roi], 
                                                           'ncsnr_threshold': [self.ROI.ncsnr_threshold_],
                                                           'model_name': [self.DNN.model_name], 
                                                           'layer': [layer], 
                                                           'domain': [domain], 
                                                           'method': [self.method],
                                                           'positive': [self.positive],
                                                           'alpha': [alpha], 
                                                           'partition': [pt], 
                                                           'cUnivar': [cUnivar], 
                                                           'cRSA': [cRSA], 
                                                           'veUnivar': [veUnivar], 
                                                           'veRSA': [veRSA]})
                                
                                self.results_df = pd.concat([self.results_df, result_row], ignore_index=True)
                                print('\tsaving')
                                self.results_df.to_csv(savefn, index=False)

                        else:
                            print('\tskipping, no features')
                                      
        return        
        
                    
def get_mapper(method, positive, alpha):
    # choose OLS or lasso
    # choose positive = True or False
    if method == 'lasso':
            mdl = Lasso(positive=positive, alpha = alpha, random_state = 365, 
                    selection = 'random', tol = 1e-2, fit_intercept=True)
    elif method == 'ols':
            mdl = LinearRegression(positive=positive, fit_intercept=True)
            
    return mdl

                                                       
def check_nan_or_inf(matrix):
    """
    Check if a matrix has any NaN or infinity values.
    
    Args:
        matrix (numpy.ndarray or array-like): Input matrix.
    
    Returns:
        bool: True if the matrix has any NaN or infinity values, False otherwise.
    """
    matrix = np.asarray(matrix)
    prop = np.mean(np.isnan(matrix)) + np.mean(np.isinf(matrix))
    valid = np.logical_and(np.logical_not(np.isnan(matrix)), 
                           np.logical_not(np.isinf(matrix)))
                                                       
    return prop, valid

                
        
        