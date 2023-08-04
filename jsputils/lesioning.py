import numpy as np
import copy
import gc
import torch
from torch import nn
from jsputils import nnutils, feature_extractor
from IPython.core.debugger import set_trace

def lesion(x,mask,apply):
    if apply is True:
        #print(f'applying mask of shape {mask.shape} to acts of shape {x.shape}')
        if len(mask.shape) == 4 and x.shape != mask.shape:
            if mask.shape[0] == x.shape[0] and mask.shape[-1] == x.shape[-1]:
                mask = torch.squeeze(mask)
            else:
                raise ValueError('mask and activation shapes are not equal')
        if torch.sum(torch.isnan(x*mask)) > 0:
            set_trace()
        return x * mask
    else:
        return x
    
def transfer_modules(from_model, to_model):
    
    # todo: refactor
    _, _, _, modules = nnutils.get_layer_names(from_model)
    
    layer_names, _ = feature_extractor.get_pretty_layer_names(from_model)
    
    for i in range(len(modules)):
        #print(layers_fmt[i].split('_')[1],modules[i])
        #setattr(to_model,layers_fmt[i].split('_')[1],modules[i])
        setattr(to_model, layer_names[i], modules[i])

    return to_model


def get_layer_dims(model, device):
    
    tmp_model = copy.deepcopy(model).eval()
    tmp_model.return_acts = True
    tmp_model.target_layers = tmp_model.layer_names

    tmp_img = torch.ones(1,3,224,224).to(device)
    _, tmp_acts = tmp_model(tmp_img)

    lay_dims = dict()
    for lay in tmp_model.target_layers:
        lay_dims[lay] = tmp_acts[lay].size()[1:]

    del tmp_model, tmp_img, tmp_acts

    torch.cuda.empty_cache()
    gc.collect()
    
    return lay_dims

def get_channelized_lesioning_masks(LSN, lay_dims, domain, method, device):
    
    print(f'applying channelized lesions for domain: {domain}')
    
    lsn_masks = dict()
    lsn_props = []
    lsn_counts = []
    
    for layer in LSN.model.layer_names:
        dims = lay_dims[layer]
        
        if method == 'relus':
            condition = 'relu' in layer
        elif method == 'full':
            condition = layer != LSN.model.layer_names[-1]
            
        if condition:
            mask = torch.from_numpy(LSN.selective_units[domain][layer]['mask'])
            mask = torch.logical_not(mask).float().view(dims)
            prop = mask.eq(0).float().mean()
            count = mask.eq(0).float().sum()
            print(f'\t\tproportion units lesioned in layer {layer}: {round(prop.item(),3)}')
            lsn_props.append(prop.item())
            lsn_counts.append(count.item())

        else:
            mask = torch.ones(lay_dims[layer])
            
        lsn_masks[layer] = mask.to(device)
        #print(layer, lsn_masks[layer].float().mean())
        
    print(f'\tsummary:\n\t\tdomain: {domain}\n\t\tmethod: {method}\n\t\tprops: {lsn_props}\n\t\tcounts: {lsn_counts}')
    print(f'\tmean prop for domain {domain}: {round(np.nanmean(lsn_props),3)}')
    print('\n\n')
        
    return lsn_masks
            

def layer_random_lesioning_mask(layer, lay_dims, p = 0.5):
    dims = lay_dims[layer]
    n_zeros = int(np.round(np.prod(dims) * p))
    mask = torch.ones(dims).flatten()
    mask[:n_zeros] = 0
    return nnutils.shuffle_tensor(mask.view(dims))

def get_random_lesioning_masks(model, lay_dims, lsn_layer, device, p):
    lsn_masks = dict()

    assert(lsn_layer in model.layer_names)
    
    for layer in model.layer_names:

        if layer == lsn_layer:
            mask = layer_random_lesioning_mask(layer, lay_dims, p)
        else:
            mask = torch.ones(lay_dims[layer])
            
        lsn_masks[layer] = mask.to(device)
        #print(layer, torch.mean(lsn_masks[layer]))

    lsn_masks['apply'] = True
    
    return lsn_masks

class LesionNet(nn.Module):
    
    def __init__(self, source_model, masks, num_classes = 1000, target_layers = None,
                return_acts = False): # default for imagenet clf
        super(LesionNet, self).__init__()
        
        self = transfer_modules(source_model, self) # transfer modules from source model

        # deal with masks
        self.masks = masks
        
        self.layer_names, _, self.layers_fmt, _ = nnutils.get_layer_names(self)
        
        if target_layers is None:
            self.target_layers = self.layer_names
        else:
            self.target_layers = target_layers
            
        self.return_acts = return_acts
        
        
    def forward(self, x):
        
        activations = dict()
        
        fc_flag = False # for knowing when to flatten (won't work for models that "widen" e.g. autoencoders)
        
        # for each layer
        for i in range(len(self.layer_names)):
            
            layer = self.layer_names[i]            

            # apply that layer's forward attribute
            operation = getattr(self,layer)
                   
            try:
                x = operation(x)
            except:
                try:
                    x = operation(x.half())
                except:
                    print(operation)
                    print(x.shape)
                    set_trace()

            # get the mask for that layer, and tile along the image dimension
            if self.masks['apply'] == True:
                if layer != self.layer_names[-1]:
                    mask = self.masks[layer].repeat(x.shape[0],1,1,1)
                else:
                    mask = self.masks[layer]
            else:
                mask = None
            
            # apply lesioning
            #x = lesion(x, mask, self.masks['apply'])
            try:
                x = lesion(x, mask, self.masks['apply'])
            except:
                print(layer)
                set_trace()
            
            if self.return_acts:
                activations[layer] = x
            
            # flatten if necessary -> do it before the first linear
            if fc_flag is False and ('fc' in self.layer_names[i+1] or 'linear' in self.layer_names[i+1]):
                x = torch.flatten(x, 1)
                fc_flag = True
                
            # helpful print statement to verify that masking worked as expected
            #print(f'# units inactive:  {torch.sum(x==0)/x.shape[0]}     ({layer})')
                
        if self.return_acts:
            return x, activations
        else:
            return x
