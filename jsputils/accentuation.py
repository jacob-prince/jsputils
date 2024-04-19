import copy
import numpy as np
import cv2
import matplotlib.pyplot as plt
import torch
from fastprogress import progress_bar

from faccent import render, param, objectives, transform
from faccent.modelzoo.util import get_model_layers
from faccent.objectives_util import _make_arg_str,_extract_act_pos,_T_handle_batch, linconv_reshape, orthogonal_proj
from faccent.objectives import wrap_objective, handle_batch

def get_drive_suppress_preds(DNN, encoding_dict, feature, input_imgs, accent_subj, pred_subj, pred_rois, layer, steps = 25, reg_lambda = 0.01, modes = ['drive','suppress'], device = 'cuda:0', plots = False):
    
    preds = dict()
    
    for roi in pred_rois:
        preds[roi] = dict()
    
    accentuations = dict()
    accentuations_resized = dict()
    
    for m, mode in enumerate(modes):
        
        if mode == 'suppress':
            steps *= 2
            reg_lambda *= 00.1

        accentuations[mode] = []
        
        for coco in progress_bar(input_imgs):
            accentuations[mode].append(accentuate_img(DNN, coco, feature = feature, mode = mode, steps = steps, reg_lambda = reg_lambda, plots = plots))
            
        #####
    
        accentuations_resized[mode] = []

        for acc in accentuations[mode]:

            img_raw = copy.deepcopy(acc[0][0].numpy().transpose(1,2,0))
            img_accent = copy.deepcopy(acc[-1][0].numpy().transpose(1,2,0))

            for c, img in enumerate([img_raw, img_accent]):

                # Resize the image to (425, 425) using OpenCV
                resized_img_float = cv2.resize(img, (425, 425), interpolation=cv2.INTER_AREA)

                # Convert the resized image to 0...255 scale and change the type to uint8
                accentuations_resized[mode].append((resized_img_float * 255).astype(np.uint8))
                
        #####
        
        all_imgs = np.stack(accentuations_resized[mode],axis=0)
        
        DNN.get_nsd_features(torch.from_numpy(all_imgs),
                                      field = f'accent')
        
        
        
        ###### 
        
        for roi in pred_rois:
            # just get the predictions to unaccentuated images the first time
            if m == 0:
                raw_pred = encoding_dict[pred_subj][roi][f'{layer}_mdl'].predict(DNN.accent[layer][::2])
                raw_pred = raw_pred - encoding_dict[pred_subj][roi]['offset']
                preds[roi]['raw'] = raw_pred

            preds[roi][f'accent_{mode}'] = encoding_dict[pred_subj][roi][f'{layer}_mdl'].predict(DNN.accent[layer][1::2])
            preds[roi][f'accent_{mode}'] = preds[roi][f'accent_{mode}'] - encoding_dict[pred_subj][roi]['offset']

    return preds, accentuations_resized


def accentuate_img(DNN, img, feature, mode = 'drive', steps = 25, reg_lambda = 0.01, plots = False, device = 'cuda:0'):
    # mask on the right: gradients from activation maximization that are large? (summed over optimization process)

    # reg lambda probably most improtant arg to mess around with 
    # if weird results w mask, try setting trans_p to 0. 
    #reg_lambda = 0.01 #0-inf: the larger this is, the more the faccent stays clsoe to original image
    trans_p = 0 #0-100: the smaller this is the more the image is whited out, except 0 means no white-out (need to change that)
    #steps = 25 #number of optimzation steps

    reg_layer = 'backbone_2'  # layer we are regularizing through,
    max_layer = 'projector_5' # this is from you, cant really mess with this without changing what directions mean

    #d1 = torch.tensor(copy.deepcopy(coef_tensor))
    d1 = torch.tensor(copy.deepcopy(feature))
    d1 = d1/d1.norm()

    # 
    parameterizer = param.fourier(init_img = img,
                                  device=device,
                                 forward_init_img=True)

    # box min size being 0.05: zooming in super close on some crops, saying, if i blow this up and feed this to the model
    # i want that little thing to also look like the direction i care about 
    # doing this in conjunction w regularization, can make things look really nice
    # "make these small areas look the same" -> can be nice, but also weird. try out by making box min size higher
    # 
    transforms = [transform.box_crop_2(
                            box_min_size=0.01, # can play with this -> more important if doing it in non-fully-connected layer
                            box_max_size=0.99,
                          ),
                  transform.uniform_gaussian_noise(),
                 ]


    #obj = objectives.channel('projector_5',d1) - .1*objectives.l2_compare('backbone_2')
    #obj_max = multi_spatial(max_layer, d1)
    obj_max = cosim(max_layer,d1) 
    obj_reg = objectives.l2_compare(reg_layer)
    
    if mode == 'drive':
        obj = (obj_max) - reg_lambda*obj_reg
    elif mode == 'suppress':
        obj = (-1*obj_max) - reg_lambda*obj_reg

    if plots:
        inline = [steps]
    else:
        inline = []
        
    imgs,losses,img_trs,img_tr_losses = render.render_vis(DNN.model,
                          obj,
                          parameterizer= parameterizer,
                          transforms = transforms,
                          out_thresholds = list(range(steps)),
                          img_tr_obj = obj_max,
                          inline_thresholds = inline,
                          trans_p = trans_p
                        )
    
    
    if plots:
        plt.figure()
        plt.plot(img_tr_losses)
        plt.show()
        
    return imgs




@wrap_objective()
def multi_spatial(layer, directions, cosine_power = 1, batch=None):
    """Visualize a cxhxw spatially weighted direction

    InceptionV1 example:
    > directions = torch.rand(512,14,14, device=device)
    > obj = objectives.multi_spatial(layer='mixed4c', directions=directions)

    Args:
        layer: Name of layer in model (string)
        directions: Direction to visualize. torch.Tensor of shape (num_channels,height,width)
        batch: Batch number (int)
        cosine_power: power to raise cosine term to

    Returns:
        Objective

    """

    @handle_batch(batch)
    def inner(model,directions = directions):
        h = model(layer)
        
        directions = directions.to(h.device)
        directions_reshape = directions
        while len(directions_reshape.shape) < 5:
            directions_reshape = directions_reshape.unsqueeze(dim=0)
            
        dot = torch.sum(directions_reshape * h, dim=2)
        
        if cosine_power != 0:
            cosine = torch.nn.CosineSimilarity(dim=2)(directions_reshape, h)
            #prevent optimizing towards negative dot and negative cosine
            thresh_cosine = torch.max(cosine,torch.tensor(.1).to(cosine.device))
            out = (thresh_cosine**cosine_power*dot)
        else:
            out = dot
            
        return -out.mean()

    return inner 

def dot_product(x, y):
    assert x.shape[2] == y.shape[0], "Mismatch in c dimension"
    
    # Handle the (t,b,c,h,w) shape case
    if len(x.shape) == 5:
        result = (x * y.unsqueeze(-1).unsqueeze(-1)).sum(dim=2)

    # Handle the (t,b,c) shape case
    elif len(x.shape) == 3:
        result = (x * y).sum(dim=2)
    
    else:
        raise ValueError("Unsupported shape for x")

    return result

# "change this image so that it points in the direction that i specified"
# but also goes far in that direction -> pretty good objective because it prevents
# from finding way to maximize 
# cosine power argument -> increase then saying you have to weigh the cosine alignment
# than the activation maximization

@wrap_objective()
def cosim(layer, direction, cosine_power = 1, batch=None, pos = None):
    """Visualize a direction as cosine x dot product

    InceptionV1 example:
    > direction = torch.rand(512, device=device)
    > obj = objectives.direction(layer='mixed4c', direction=direction)

    Args:
        layer: Name of layer in model (string)
        direction: Direction to visualize. torch.Tensor of shape (num_channels,)
        batch: Batch number (int)
        pos: hxw position in activation map to maximize (tuple)
        cosine_power: power to raise cosine term to

    Returns:
        Objective

    """

    @handle_batch(batch)
    def inner(model,direction = direction):
        h = model(layer)
        direction = direction.to(h.device)
        if len(h.shape) == 3:  #linear layer
            direction_reshape = direction.reshape((1, 1, -1))
        else:
            direction_reshape = direction.reshape((1, 1, -1, 1, 1))

        cosine = torch.nn.CosineSimilarity(dim=2)(direction_reshape, h)
        #prevent optimizing towards negative dot and negative cosine
        thresh_cosine = torch.max(cosine,torch.tensor(.1).to(cosine.device))
        dot = dot_product(h, direction)
        out = (thresh_cosine**cosine_power*dot)
        if pos is not None:
            out = out = out[:,:, pos[0], pos[1]]
        return -out.mean()

    return inner   
