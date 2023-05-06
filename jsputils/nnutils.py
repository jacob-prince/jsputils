import os
from os.path import exists
import torch
from torch import nn
import torchvision
from torchvision import transforms
import torchlens as tl
import numpy as np
import copy
import pandas as pd
import gc
import scipy.stats as stats
from fastprogress import progress_bar
from IPython.core.debugger import set_trace

from torchvision.transforms._presets import ImageClassification

from jsputils import paths, models

def load_model(model_name):
    
    if 'alexnet-supervised' in model_name:
        
        if 'random' in model_name: 
            model = torchvision.models.alexnet()
        else:
            # switch to eval mode
            model = torchvision.models.alexnet(weights='DEFAULT')

        transform = ImageClassification(crop_size=224)
        
        # todo: find where this is
        state_dict = None
        
        is_categ_supervised = True
        
        model_str = model_name
        
    elif 'alexnet-barlow-twins' in model_name:
        
        if 'random' in model_name:
            model, state_dict = models.alexnet_gn_barlow_twins(pretrained=False)
        else:
            model, state_dict = models.alexnet_gn_barlow_twins(pretrained=True)
            
        transform = ImageClassification(resize_size=224, crop_size=224)
        
        is_categ_supervised = False
        
        model_str = model_name
        
    elif 'alexnet-vggface' in model_name:
        
        checkpoint = torch.load(f'{paths.weight_savedir()}/alexnet_faces_final.pth.tar',map_location='cpu')
        state_dict = checkpoint['state_dict']
    
        state_dict = {str.replace(k,'module.',''): v for k,v in state_dict.items()}
        model = torchvision.models.alexnet(num_classes=3372)
        model.load_state_dict(state_dict)
        
        transform = ImageClassification(resize_size=224, crop_size=224)
        
        is_categ_supervised = True
        
        model_str = model_name
        
    elif 'dropout' in model_name:
        
        # todo: refactor
        sweep = 'sweep1'
        dropout_prop = float(model_name.split('dropout-')[-1])
        learning_rate = 0.05
        lr_peak = 0.15
        
        model, transform, state_dict, is_categ_supervised, model_str = load_alexnet_dropout_model(sweep, 
                                                                                                  dropout_prop, 
                                                                                                  learning_rate,
                                                                                                  lr_peak)
        
    # ensure that all relus are converted to inplace=False
    convert_relu(model.eval())
        
    return model, transform, state_dict, is_categ_supervised, model_str

def load_alexnet_dropout_model(sweep, dropout_prop, learning_rate, lr_peak):
    
    weight_dir = paths.dnn_dropout_weightdir()
    
    model_list = os.listdir(f'{weight_dir}/{sweep}')
    
    dropout_props = np.unique([float(mdl.split('_')[1][4:])/100 for mdl in model_list])
    learning_rates = np.unique([float(mdl.split('_')[3][3:])/100 for mdl in model_list])
    lr_peaks = np.unique([float(mdl.split('_')[4][4:])/100 for mdl in model_list])
    
    if sweep == 'sweep1':
        model_str = f'alexnet_drop{int(100*dropout_prop):03}_ep100_lr{int(100*learning_rate):02}_peak{int(100*lr_peak):02}_ramp65-76'
        is_categ_supervised = True
    else:
        raise NotImplementedError()
    
    this_model_dir = f'{weight_dir}/{sweep}/{model_str}'
    
    weight_fn = f'{this_model_dir}/{os.listdir(this_model_dir)[0]}/final_weights.pt'
    assert(exists(weight_fn))

    model = models.AlexNet(dropout = dropout_prop).eval()
    
    state_dict = torch.load(weight_fn)
    state_dict = {str.replace(k,'module.',''): v for k,v in state_dict.items()}

    model.load_state_dict(state_dict)
    
    transform = ImageClassification(crop_size=224)
    
    return model, transform, state_dict, is_categ_supervised, model_str

def get_layer_names_and_dims(model):
    
    x = torch.rand(1, 3, 224, 224)
    
    model_history = tl.get_model_activations(copy.deepcopy(model.eval()), x, which_layers='all')
    
    layer_list = model_history.layer_labels
    
    layer_list = np.array([lay for lay in layer_list if not 'buffer' in lay and not 'input' in lay])
    
    layer_dims = dict()
    
    for layer in layer_list:
        layer_dims[layer] = model_history[layer].tensor_contents.detach().numpy().shape[1:]
    
    del model_history
    gc.collect()
    
    return layer_list, layer_dims

def shuffle_tensor(tensor):
    return tensor.flatten()[torch.randperm(tensor.numel())].view(tensor.size())
    
#######################
    
def get_NSD_train_test_activations(model_name, image_data):
    
    model, transform, _ = load_model(model_name)
    
    activations = dict()

    for partition in ['train','test']:

        activations[partition] = dict()

        # transform the images
        X = transform(torch.from_numpy(image_data[partition].transpose(0,3,1,2)))

        model_history = tl.get_model_activations(model, X, which_layers='all')

        for layer in progress_bar(model_history.layer_labels):
            activations[partition][layer] = model_history[layer].tensor_contents.detach().numpy()
            
    return activations 

def alexnet_layer_str_format(layer_list):
    
    out = []
    c = 0
    for layer in layer_list:
        if 'conv2d' in layer:
            c+=1
            out.append(f'conv{c}')
        elif 'relu' in layer:
            out.append(f'relu{c}')
        elif 'maxpool' in layer:
            out.append(f'maxpool{c}')
        elif 'groupnorm' in layer:
            out.append(f'groupnorm{c}')
        elif 'batchnorm' in layer:
            out.append(f'batchnorm{c}')
        elif 'linear' in layer:
            c+=1
            out.append(f'fc{c}')
            
    assert(len(out) == len(layer_list))
    return out


def get_layer_group(model_name, layer_list = []):
    
    if 'alexnet-supervised' in model_name or 'alexnet-vggface' in model_name:
        
        all_layer_names = ['conv2d_1_2', 'relu_1_3', 'maxpool2d_1_4', 
                           'conv2d_2_5', 'relu_2_6', 'maxpool2d_2_7', 
                           'conv2d_3_8', 'relu_3_9', 
                           'conv2d_4_10', 'relu_4_11', 
                           'conv2d_5_12', 'relu_5_13', 'maxpool2d_3_14', 
                           'linear_1_19', 'relu_6_20', 
                           'linear_2_23', 'relu_7_24', 
                           'linear_3_25']
        
    elif 'alexnet_drop' in model_name:
        
        all_layer_names = ['conv2d_1_2',
                         'relu_1_3',
                         'maxpool2d_1_4',
                         'conv2d_2_5',
                         'relu_2_6',
                         'maxpool2d_2_7',
                         'conv2d_3_8',
                         'relu_3_9',
                         'conv2d_4_10',
                         'relu_4_11',
                         'conv2d_5_12',
                         'relu_5_13',
                         'maxpool2d_3_14',
                         'linear_1_18',
                         'relu_6_19',
                         'linear_2_21',
                         'relu_7_22',
                         'linear_3_23']
        
    elif 'alexnet-barlow-twins' in model_name:
        all_layer_names = ['conv2d_1_8',
                            'groupnorm_1_9',
                             'relu_1_10',
                             'maxpool2d_1_11',
                             'conv2d_2_12',
                             'groupnorm_2_13',
                             'relu_2_14',
                             'maxpool2d_2_15',
                             'conv2d_3_16',
                             'groupnorm_3_17',
                             'relu_3_18',
                             'conv2d_4_19',
                             'groupnorm_4_20',
                             'relu_4_21',
                             'conv2d_5_22',
                             'groupnorm_5_23',
                             'relu_5_24',
                             'maxpool2d_3_25',
                             'linear_1_28',
                             'batchnorm_1_29',
                             'relu_6_30',
                             'linear_2_31',
                             'batchnorm_2_32',
                             'relu_7_33',
                             'linear_3_34',
                             'batchnorm_3_35']

    if layer_list == []:
        layers_to_analyze = all_layer_names
    else:
        for lay in layer_list:
            assert(lay in all_layer_names)
        layers_to_analyze = layer_list
        
    return layers_to_analyze

def convert_relu(parent):
    for child_name, child in parent.named_children():
        if isinstance(child, nn.ReLU):
            setattr(parent, child_name, nn.ReLU(inplace=False))
        elif len(list(child.children())) > 0:
            convert_relu(child)
            
def get_modules(model):
    
    lay_modules = []
    
    layers = list(model.named_modules())
    
    count = 1
    conv_count = 0

    for i in range(1,len(layers)):
        
        module_type = layers[i][1].__class__.__name__

        nonbasic_types = np.array(['Sequential','BasicBlock','Bottleneck','Fire',
                                 '_DenseBlock', '_DenseLayer', 'Transition', '_Transition','InvertedResidual','_InvertedResidual','ConvBNReLU','CORblock_Z','CORblock_S','CORblock'])

        if np.logical_not(np.isin(module_type, nonbasic_types)):

            lay_modules.append(layers[i][1])

    return lay_modules
                        
            
### todo: refactor/eliminate need for this

def get_layer_names(model):
    
    lay_names_torch = []
    lay_names_user = []
    lay_conv_strs = []
    lay_modules = []
    
    layers = list(model.named_modules())
    
    count = 1
    conv_count = 0

    for i in range(1,len(layers)):
        
        module_type = layers[i][1].__class__.__name__

        nonbasic_types = np.array(['Sequential','BasicBlock','Bottleneck','Fire',
                                 '_DenseBlock', '_DenseLayer', 'Transition', '_Transition','InvertedResidual','_InvertedResidual','ConvBNReLU','CORblock_Z','CORblock_S','CORblock'])

        conv_types = ['Conv','Linear']

        if np.logical_not(np.isin(module_type, nonbasic_types)):

            # get layer naming info
            lay_names_torch.append(layers[i][0])
            lay_names_user.append(str(count) + '_' + str(layers[i][1]).split('(')[0])

            # get conv tag
            if any(s in lay_names_user[-1] for s in conv_types) and 'downsample' not in lay_names_torch[-1]:
                conv_count += 1
            lay_conv_strs.append(str(conv_count))

            # update
            lay_names_user[-1] = lay_names_user[-1] + '_' + lay_conv_strs[-1]
            
            lay_modules.append(layers[i][1])

            count += 1
    
    lay_names_user_fmt = []
    
    c = 0
    for lay in lay_names_user:
        lay_type = lay.split('_')[1]
        
        if 'Conv' in lay_type:
            if 'downsample' in lay_names_torch[c]:
                fmt = 'downsample'
            elif 'skip' in lay_names_torch[c]:
                fmt = 'conv_skip'
            else:
                fmt = 'conv'
        elif 'Norm' in lay_type:
            if 'skip' in lay_names_torch[c]:
                fmt = 'norm_skip'
            else:
                fmt = 'norm'
        elif 'ReLU' in lay_type:
            fmt = 'relu'
        elif 'MaxPool' in lay_type:
            fmt = 'maxpool'
        elif 'AvgPool' in lay_type:
            fmt = 'avgpool'
        elif 'Linear' in lay_type:
            fmt = 'fc'
        elif 'Dropout' in lay_type:
            fmt = 'drop'
        elif 'Identity' in lay_type:
            fmt = 'identity'
        elif 'Flatten' in lay_type:
            fmt = 'flatten'
        elif 'Lesion' in lay_type:
            fmt = 'lesion'
        elif 'Sigmoid' in lay_type:
            fmt = 'sigmoid'
        elif 'Inception' in lay_type:
            fmt = 'inception'
        elif 'SelectAdaptivePool2d' in lay_type:
            fmt = 'adaptivepool'
        else:
            print(lay_type)
            raise ValueError('fmt not implemented yet')
        c+=1
        
        lay_names_user_fmt.append(lay.split('_')[0] + '_' + fmt + lay.split('_')[2])

    #print(lay_names_user_fmt)
        
    return lay_names_torch, lay_names_user, lay_names_user_fmt, lay_modules