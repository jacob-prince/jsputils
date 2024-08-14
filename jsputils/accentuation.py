import copy
import numpy as np
import cv2
import matplotlib.pyplot as plt
import torch
from fastprogress import progress_bar
from PIL import Image
from os.path import exists, join
import os
from IPython.core.debugger import set_trace

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


def accentuate_img(DNN, img, feature, mode = 'drive', steps = 20, reg_lambda = 0.01, plots = False, device = 'cuda:0'):
    # mask on the right: gradients from activation maximization that are large? (summed over optimization process)

    # reg lambda probably most improtant arg to mess around with 
    # if weird results w mask, try setting trans_p to 0. 
    #reg_lambda = 0.01 #0-inf: the larger this is, the more the faccent stays clsoe to original image
    trans_p = 0 #0-100: the smaller this is the more the image is whited out, except 0 means no white-out (need to change that)
    #steps = 25 #number of optimzation steps

    reg_layer = 'backbone_0'  # layer we are regularizing through,
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
                            box_min_size=0.005, # can play with this -> more important if doing it in non-fully-connected layer
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
                          trans_p = trans_p,
                                        
                          
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

def calculate_luminance_contrast(image_array: np.ndarray):
    """
    Calculates the mean luminance and contrast of an image.
    
    Args:
    image_array (np.ndarray): The input image as a numpy array.
    
    Returns:
    tuple: mean luminance, contrast (standard deviation of luminance)
    """
    # Convert to YCbCr color space to separate luminance
    ycbcr = Image.fromarray(image_array).convert('YCbCr')
    y, _, _ = ycbcr.split()
    
    # Convert luminance to numpy array
    luminance = np.array(y)
    
    # Calculate mean luminance
    mean_luminance = luminance.mean()
    
    # Calculate contrast (standard deviation of luminance)
    contrast = luminance.std()
    
    return mean_luminance, contrast

def match_luminance_contrast(image_array: np.ndarray, target_luminance: float, target_contrast: float):
    """
    Adjusts the luminance and contrast of an image to match the target values.
    
    Args:
    image_array (np.ndarray): The input image as a numpy array.
    target_luminance (float): The target mean luminance.
    target_contrast (float): The target contrast (standard deviation of luminance).
    
    Returns:
    np.ndarray: The adjusted image as a numpy array.
    """
    # Convert to YCbCr color space to separate luminance
    ycbcr = Image.fromarray(image_array).convert('YCbCr')
    y, cb, cr = ycbcr.split()
    
    # Convert luminance to numpy array
    luminance = np.array(y)
    
    # Calculate current mean luminance and contrast
    current_luminance, current_contrast = calculate_luminance_contrast(image_array)
    
    # Adjust contrast
    if current_contrast != 0:
        scale = target_contrast / current_contrast
    else:
        scale = 1
    adjusted_luminance = (luminance - current_luminance) * scale + current_luminance
    
    # Adjust luminance
    adjusted_luminance += (target_luminance - current_luminance)
    
    # Clip to valid range
    adjusted_luminance = np.clip(adjusted_luminance, 0, 255).astype(np.uint8)
    
    # Merge adjusted luminance with original chrominance channels
    adjusted_ycbcr = Image.merge('YCbCr', (Image.fromarray(adjusted_luminance), cb, cr))
    
    # Convert back to RGB
    adjusted_rgb = adjusted_ycbcr.convert('RGB')
    
    return np.array(adjusted_rgb)


def make_square_and_resize(image_array: np.ndarray, size: int = 425) -> np.ndarray:
    """
    Pads the given numpy array image to make it square by adding white padding,
    then resizes it to the specified size.
    
    Args:
    image_array (np.ndarray): The input image as a numpy array.
    size (int): The size to resize the image to (default is 425).
    
    Returns:
    np.ndarray: The resulting image as a numpy array.
    """
    # Convert numpy array to PIL image
    image = Image.fromarray(image_array)
    
    # Get the dimensions of the image
    width, height = image.size
    
    # Determine the size of the new square image
    new_size = max(width, height)
    
    # Calculate the padding values
    padding_left = (new_size - width) // 2
    padding_right = new_size - width - padding_left
    padding_top = (new_size - height) // 2
    padding_bottom = new_size - height - padding_top
    
    # Create a new image with white background
    new_image = Image.new("RGB", (new_size, new_size), (255, 255, 255))
    
    # Paste the original image onto the center of the new image
    new_image.paste(image, (padding_left, padding_top))
    
    # Resize the image to the specified size
    resized_image = new_image.resize((size, size), Image.ANTIALIAS)
    
    # Convert the resized PIL image back to a numpy array
    resized_image_array = np.array(resized_image)
    
    return resized_image_array

def process_directory(input_dir: str, output_dir: str):
    """
    Processes all images in the input directory to match their luminance and contrast,
    and saves them to the output directory.
    
    Args:
    input_dir (str): Path to the input directory containing folders of images.
    output_dir (str): Path to the output directory where processed images will be saved.
    """
    # Create the output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    image_arrays = []
    file_paths = []
    
    # Traverse the input directory and load images
    for root, dirs, files in os.walk(input_dir):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                file_path = os.path.join(root, file)
                #print(f"Loading image: {file_path}")
                image_array = np.array(Image.open(file_path).convert('RGB'))
                image_arrays.append(image_array)
                file_paths.append(file_path)
    
    if not image_arrays:
        print("No images found in the input directory.")
        return
    
    # Calculate mean luminance and contrast for each image
    luminance_list = []
    contrast_list = []
    
    for image_array in image_arrays:
        luminance, contrast = calculate_luminance_contrast(image_array)
        luminance_list.append(luminance)
        contrast_list.append(contrast)
    
    # Calculate target luminance and contrast (mean of all images)
    target_luminance = np.mean(luminance_list)
    target_contrast = np.mean(contrast_list) + 0.25*np.std(contrast_list)
    
    #print(f"Target luminance: {target_luminance}, Target contrast: {target_contrast}")
    
    # Process each image to match the target luminance and contrast
    for image_array, file_path in zip(image_arrays, file_paths):
        adjusted_image_array = match_luminance_contrast(image_array, target_luminance, target_contrast)
        
        # Make the image square and resize it to 425x425 pixels
        final_image_array = make_square_and_resize(adjusted_image_array, size=425)
        
        # Determine the output file path
        relative_path = os.path.relpath(file_path, input_dir)
        output_file_path = os.path.join(output_dir, relative_path)
        
        # Create the output directory if it doesn't exist
        os.makedirs(os.path.dirname(output_file_path), exist_ok=True)
        
        print(f"Saving processed image: {output_file_path}")
        
        # Save the adjusted image
        Image.fromarray(final_image_array).save(output_file_path, 'JPEG')
        
    print("Processing complete.")

def verify_luminance_contrast(output_dir: str, tolerance: float = 1):
    """
    Verifies that the luminance and contrast are matched among all images
    in the output directory.
    
    Args:
    output_dir (str): Path to the output directory containing processed images.
    tolerance (float): Tolerance for floating-point comparison.
    """
    luminance_list = []
    contrast_list = []
    
    # Traverse the output directory and load images
    for root, dirs, files in os.walk(output_dir):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                file_path = os.path.join(root, file)
                #print(f"Verifying image: {file_path}")
                image_array = np.array(Image.open(file_path).convert('RGB'))
                luminance, contrast = calculate_luminance_contrast(image_array)
                luminance_list.append(luminance)
                contrast_list.append(contrast)
    
    # Assert that all luminance and contrast values are identical within the tolerance
    set_trace()
    target_luminance = luminance_list[0]
    target_contrast = contrast_list[0]
    
    for luminance in luminance_list:
        assert abs(luminance - target_luminance) <= tolerance, f"Luminance mismatch: {luminance} != {target_luminance}"
    
    for contrast in contrast_list:
        assert abs(contrast - target_contrast) <= tolerance, f"Contrast mismatch: {contrast} != {target_contrast}"
    
    print(f"Mean luminance: {target_luminance}, Mean contrast: {target_contrast}")
    print("All images have identical luminance and contrast within the tolerance.")
    print("Verification complete.")


    
    
def get_lumiance(image_path):
    image = cv2.imread(image_path)
    # Convert to LAB color space
    lab_image = cv2.cvtColor(image, cv2.COLOR_BGR2Lab)
    # Split the LAB image to L, A, and B channels
    l, a, b = cv2.split(lab_image)
    # Normalize the L channel
    l_float = l.astype(np.float32)  # Convert L to float for processing
    thisMean = np.mean(l_float)
    return thisMean

def get_contrast(image_path):
    image = cv2.imread(image_path)
    # Convert to LAB color space
    lab_image = cv2.cvtColor(image, cv2.COLOR_BGR2Lab)
    # Split the LAB image to L, A, and B channels
    l, a, b = cv2.split(lab_image)
    # Normalize the L channel
    l_float = l.astype(np.float32)  # Convert L to float for processing
    thisStdev = np.std(l_float)
    return thisStdev

def normalize_luminance(image_path, output_directory):
    # Read the image
    image = cv2.imread(image_path)
    # Convert to LAB color space
    lab_image = cv2.cvtColor(image, cv2.COLOR_BGR2Lab)
    # Split the LAB image to L, A, and B channels
    l, a, b = cv2.split(lab_image)
    # Normalize the L channel
    l_float = l.astype(np.float32)  # Convert L to float for processing
    thisMean = np.mean(l_float)
    thisStdev = np.std(l_float)
    normalized_l = ((l_float - thisMean) * (32 / thisStdev)) + 128
    normalized_l = np.clip(normalized_l, 0, 255).astype(np.uint8)  # Clip values to [0, 255] and convert back to uint8
    # Merge the LAB channels back together
    normalized_lab = cv2.merge([normalized_l, a, b])
    # Convert back to RGB
    normalized_rgb = cv2.cvtColor(normalized_lab, cv2.COLOR_Lab2BGR)
    # Save the image
    filename = os.path.basename(image_path)
    output_path = os.path.join(output_directory,filename)
    cv2.imwrite(output_path, normalized_rgb)
    