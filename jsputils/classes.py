import torch
from os.path import exists
from torch import nn
import torchvision
import copy
import PIL.Image as Image
import matplotlib.pyplot as plt
import numpy as np
import os
from jsputils import paths, nsdorg, plotting, encoding, nnutils, selectivity, validation, lesioning, feature_extractor
from fastprogress import progress_bar
import gc
from IPython.core.debugger import set_trace

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
            self.domain_colors = np.array(['red','dodgerblue','orange','limegreen','blueviolet','lightgray'])
            self.subdir_domain_ref = np.array([0,1,2,0,3,3,2,1,4,5,4])
            
        elif image_set_name == 'classic-categ':
            self.categ_nimg = 80
            self.floc_subdirs = np.array(['1-Faces','2-Bodies','3-Scenes','4-Words',
                                          '5-Objects','6-ScrambledObjects','7-ScrambledWords'])
            self.floc_domains = np.array(['faces','bodies','objects','scenes','characters','scrambled'])
            self.domain_colors = np.array(['red','dodgerblue','orange','limegreen','blueviolet','lightgray'])
            self.subdir_domain_ref = np.array([0,1,2,3,4,5,5])
            
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
    def __init__(self, partition, device = 'cuda:0', batch_size = 512, num_workers = 64):
        self.partition = partition
        self.device = device
        self.batch_size = batch_size
        self.num_workers = num_workers
           
        if self.partition == 'train':
            pass
        elif self.partition == 'val':
            self.data_loader = validation.create_val_loader(paths.ffcv_imagenet1k_valset(), 
                                                            self.device,
                                                            self.num_workers, 
                                                            self.batch_size)
            
class DataLoaderTorch:
    def __init__(self, dataset, batch_size = 512, num_workers = 4, shuffle = False, pin_memory = False):
        
        self.dataset = dataset
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.shuffle = shuffle
        self.pin_memory = pin_memory
        
        self.data_loader = torch.utils.data.DataLoader(
            dataset=self.dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=self.shuffle,
            pin_memory=self.pin_memory,
        )

        

class DNNModel:
    def __init__(self, model_name):
                         
        self.model, self.transforms, self.state_dict, self.is_categ_supervised, self.model_name = nnutils.load_model(model_name)
        
        self.layer_names_fmt, self.layer_names_torch = feature_extractor.get_pretty_layer_names(self.model)
                 
        # self.layer_names, self.layer_dims = nnutils.get_layer_names_and_dims(self.model)
        # self.formatted_layer_names = None
        
    def get_floc_features(self, ImageSet, field = 'floc_features', device = 'cuda:0'):
        
        if not hasattr(self, field):
            #for imgs in data_loader
            data_loader = torch.utils.data.DataLoader(
                        dataset=ImageSet.images,
                        batch_size=len(ImageSet.images),
                        num_workers=1,
                        shuffle=False,
                        pin_memory=False
                    )

            image_tensors, _ = next(iter(data_loader))

            setattr(self, field, feature_extractor.get_features(copy.deepcopy(self.model), 
                            image_tensors, self.layer_names_torch, self.layer_names_fmt, device = device))

            del image_tensors, data_loader
            torch.cuda.empty_cache()
            gc.collect()
        else:
            print(f'skipping activation extraction: attribute {field} already exists in DNN.')

    def find_selective_units(self, localizer_image_set, FDR_p = 0.05, overwrite = False, verbose = False):
        
        # Define the indices of category-selective units using localizer images
        floc_imageset = ImageSet(localizer_image_set, transforms = self.transforms)
        
        self.selective_units = selectivity.run_dnn_localizer_procedure(self, floc_imageset, FDR_p, overwrite, verbose)
        
    def get_imagenet_accs(self, topk = 5): 
        
        ValLoader = DataLoaderFFCV('val')
        
        self.imagenet_accs = validation.get_imagenet_class_accuracies(copy.deepcopy(self.model).half().to(ValLoader.device),
                                                                      ValLoader.data_loader,
                                                                      ValLoader.device, topk = topk)
            
            
    def train_linear_probe(self):
        # Train a linear ImageNet probe for self-supervised models
        pass

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
        
    def remove_randomized_lesions(self):
        masks = dict()
        masks['apply'] = False
        self.model.masks = masks
                
    def get_imagenet_accs(self, topk = 5): 
        
        ValLoader = DataLoaderFFCV('val')
        
        self.imagenet_accs = validation.get_imagenet_class_accuracies(copy.deepcopy(self.model).half().to(ValLoader.device),
                                                                      ValLoader.data_loader,
                                                                      ValLoader.device, topk = topk)
    
    def get_selective_unit_acts(self):
        
        assert(hasattr(self, 'selective_units'))
            
        ValLoader = DataLoaderFFCV('val')
        
        print(self.model.layer_names)
        
        self.selective_unit_acts = validation.get_selective_unit_acts(copy.deepcopy(self.model).half().to(ValLoader.device),
                                                                      copy.deepcopy(self.selective_units),
                                                                      list(self.selective_units['faces'].keys()),
                                                                      ValLoader.data_loader,
                                                                      ValLoader.device)

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
    def __init__(self, subj, roi, ncsnr_threshold, plot = True):
        
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
            
    def load_train_test_data(self, train_imageset, test_imageset):
        print('loading train and test data')
        self.image_data, self.brain_data = nsdorg.get_NSD_train_test_images_and_betas(subj = self.subj,
                                                                                      space = self.space,
                                                                                      subj_betas = self.betas,
                                                                                      rep_cocos = self.rep_cocos,
                                                                                      train_imageset = train_imageset,
                                                                                      test_imageset = test_imageset,
                                                                                      mean = False)
    def get_ncsnr_mask(self, threshold):
        self.ncsnr_threshold_ = threshold
        self.mask = self.ncsnr[self.included_voxel_idx['full']] > self.ncsnr_threshold_

    def save(self):
        np.save(self.savefn, self, allow_pickle=True)
           