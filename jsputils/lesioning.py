import torch
from torch import nn
from jsputils import nnutils
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
    
    _, _, layers_fmt, modules = nnutils.get_layer_names(from_model)
    
    for i in range(len(modules)):
        #print(layers_fmt[i].split('_')[1],modules[i])
        setattr(to_model,layers_fmt[i].split('_')[1],modules[i])

    return to_model

class LesionNet(nn.Module):
    
    def __init__(self, source_model, masks, num_classes = 1000, target_layers = None,
                return_acts = True): # default for imagenet clf
        super(LesionNet, self).__init__()
        
        self = transfer_modules(source_model, self) # transfer modules from source model
    
        # deal with masks
        self.masks = masks
        
        self.layers, _, self.layers_fmt, _ = nnutils.get_layer_names(self)
        
        if target_layers is None:
            self.target_layers = self.layers
        else:
            self.target_layers = target_layers
            
        self.return_acts = return_acts
        
        
    def forward(self, x):
        
        activations = dict()
        
        fc_flag = False # for knowing when to flatten (won't work for models that "widen" e.g. autoencoders)
        
        # for each layer
        for i in range(len(self.layers)):
            
            layer = self.layers[i]            

            # apply that layer's forward attribute
            operation = getattr(self,layer)
                   
            try:
                x = operation(x)
            except:
                print(operation)
                print(x.shape)
                set_trace()

            # get the mask for that layer, and tile along the image dimension
            if self.masks['apply'] == True:
                if layer != self.layers[-1]:
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
            if fc_flag is False and ('fc' in self.layers[i+1] or 'linear' in self.layers[i+1]):
                x = torch.flatten(x, 1)
                fc_flag = True
                
            # helpful print statement to verify that masking worked as expected
            #print(f'# units inactive:  {torch.sum(x==0)/x.shape[0]}     ({layer})')
                
        if self.return_acts:
            return x, activations
        else:
            return x
