import torch
import torch.nn as nn
from torchvision import models
from pdb import set_trace
import gc

class FeatureExtractor(nn.Module):
    '''
        FeatureExtractor class that allows you to retain outputs of any layer.
        This class uses PyTorch's "forward hooks", which let you insert a function
        that takes the input and output of a module as arguements.
        In this hook function you can insert tasks like storing the intermediate values,
        or as we'll do in the FeatureEditor class, actually modify the outputs.
        Adding these hooks can cause headaches if you don't "remove" them 
        after you are done with them. For this reason, the FeatureExtractor is 
        setup to be used as a context, which sets up the hooks when
        you enter the context, and removes them when you leave:

        layer_names = [...]
        with FeatureExtractor(model, layer_names) as extractor:
            features = extractor(imgs)

        If there's an error in that context (or you cancel the operation),
        the __exit__ function of the feature extractor is executed,
        which we've setup to remove the hooks. This will save you 
        headaches during debugging/development.
    '''    
    def __init__(self, model, layers, retain=True, detach=True, clone=True, device='cpu'):
        super().__init__()
        layers = [layers] if isinstance(layers, str) else layers
        self.model = model
        self.layers = layers
        self.detach = detach
        self.clone = clone
        self.device = device
        self.retain = retain
        self._features = {layer: torch.empty(0) for layer in layers}        
        self.hooks = {}
        
    def hook_layers(self):        
        self.remove_hooks()
        for layer_id in self.layers:
            layer = dict([*self.model.named_modules()])[layer_id]

            self.hooks[layer_id] = layer.register_forward_hook(self.save_outputs_hook(layer_id))
    
    def remove_hooks(self):
        for layer_id in self.layers:
            if self.retain == False:
                self._features[layer_id] = torch.empty(0)
            if layer_id in self.hooks:
                self.hooks[layer_id].remove()
                del self.hooks[layer_id]
    
    def __enter__(self, *args): 
        self.hook_layers()
        return self
    
    def __exit__(self, *args): 
        self.remove_hooks()
            
    def save_outputs_hook(self, layer_id):
        def detach(output):
            if isinstance(output, tuple): return tuple([o.detach() for o in output])
            elif isinstance(output, list): return [o.detach() for o in output]
            else: return output.detach()
        def clone(output):
            if isinstance(output, tuple): return tuple([o.clone() for o in output])
            elif isinstance(output, list): return [o.clone() for o in output]
            else: return output.clone()
        def to_device(output, device):
            if isinstance(output, tuple): return tuple([o.to(device) for o in output])
            elif isinstance(output, list): return [o.to(device) for o in output]
            else: return output.to(device)
        def fn(_, __, output):
            if self.detach: output = detach(output)
            if self.clone: output = clone(output)
            if self.device: output = to_device(output, self.device)
            self._features[layer_id] = output
        return fn

    def forward(self, x, *args, **kwargs):
        _ = self.model(x, *args, **kwargs)
        return self._features
    
def get_pretty_layer_names(model):
    fmt = []
    c = 1
    module_strs = get_module_names(model)
    for m, module in enumerate(module_strs):
        if 'conv' in module.lower():
            if m > 0:
                if 'dropout' not in module_strs[m-1].lower():
                    c+=1
            fmt.append(f'conv{c}')
        elif 'linear' in module.lower():
            if m > 0:
                if 'dropout' not in module_strs[m-1].lower():
                    c+=1
            fmt.append(f'fc{c}')
        elif 'relu' in module.lower():
            fmt.append(f'relu{c}')
        elif 'maxpool' in module.lower():
            fmt.append(f'maxpool{c}')
        elif 'avgpool' in module.lower():
            fmt.append(f'avgpool{c}')
        elif 'groupnorm' in module.lower():
            fmt.append(f'groupnorm{c}')
        elif 'batchnorm' in module.lower():
            fmt.append(f'batchnorm{c}')
        elif 'flatten' in module.lower():
            fmt.append(f'flatten')
        elif 'dropout' in module.lower():
            c+=1 
            fmt.append(f'dropout{c}')
        elif 'normalize' in module.lower():
            fmt.append(f'norm{c}')
        else:
            raise NotImplementedError(f'format for module {module} not implemented yet')
    return fmt, get_layer_names(model)
        
def get_module_names(model):
    return get_modules(model, parent_name='', module_info=[])

def get_modules(model, parent_name='', module_info=[]):
    for module_name, module in model.named_children():
        layer_name = parent_name + '.' + module_name
        if len(list(module.named_children())):
            module_info = get_modules(module, layer_name, module_info=module_info)
        else:
            module_info.append(str(module).split('(')[0])
            
    return module_info

def get_layers(model, parent_name='', layer_info=[]):
    for module_name, module in model.named_children():
        layer_name = parent_name + '.' + module_name
        if len(list(module.named_children())):
            layer_info = get_layers(module, layer_name, layer_info=layer_info)
        else:
            layer_info.append(layer_name.strip('.'))
    
    return layer_info

def get_layer_names(model):
    return get_layers(model, parent_name='', layer_info=[])

def get_layer_type(model, layer_name):
    for name,m in list(model.named_modules()):
        if name == layer_name: return m.__class__.__name__
            
def convert_relu_layers(parent):
    for child_name, child in parent.named_children():
        if isinstance(child, nn.ReLU) and child.inplace==True:
            setattr(parent, child_name, nn.ReLU(inplace=False))
        elif len(list(child.children())) > 0:
            convert_relu_layers(child)
            
def get_layer_shapes(model, layer_names, x):
    model.eval()
    with torch.no_grad():        
        with FeatureExtractor(model, layer_names) as extractor:
            features = extractor(x)
            shapes = {k:v.shape for k,v in features.items()}
    return shapes

def get_features(model, images, in_layers, out_layers, device, lesion_net = False):
    model.eval()
    with torch.no_grad():
        
        with FeatureExtractor(model.to(device), in_layers) as extractor:
            features = extractor(images.to(device))
            
    out = dict()
    for i, k in enumerate(features):
        out[out_layers[i]] = features[k].detach().cpu().numpy()
        
    del images, extractor, features
    torch.cuda.empty_cache()
    gc.collect()
    
    return out