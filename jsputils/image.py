import os
import cv2
import numpy as np

def calculate_luminance_contrast(image):
    lab_image = cv2.cvtColor(image, cv2.COLOR_BGR2Lab)
    l, _, _ = cv2.split(lab_image)
    l_float = l.astype(np.float32)
    mean_luminance = np.mean(l_float)
    contrast = np.std(l_float)
    return mean_luminance, contrast

def normalize_luminance(image_path, output_directory, target_luminance, target_contrast):
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Image at path {image_path} could not be read.")
    
    # Convert to LAB color space
    lab_image = cv2.cvtColor(image, cv2.COLOR_BGR2Lab)
    l, a, b = cv2.split(lab_image)
    l_float = l.astype(np.float32)
    
    # Calculate current mean luminance and contrast
    this_mean = np.mean(l_float)
    this_stdev = np.std(l_float)
    print(image_path)
    print(f"Initial luminance: {this_mean}, Initial contrast: {this_stdev}")
    
    # Adjust contrast
    normalized_l = ((l_float - this_mean) * (target_contrast / this_stdev)) + target_luminance
    print(f"Luminance after contrast adjustment: {np.mean(normalized_l)}, Contrast after adjustment: {np.std(normalized_l)}")
    
    # Clip to valid range
    normalized_l = np.clip(normalized_l, 0, 255).astype(np.uint8)
    
    # Merge adjusted luminance with original chrominance channels
    normalized_lab = cv2.merge([normalized_l, a, b])
    normalized_rgb = cv2.cvtColor(normalized_lab, cv2.COLOR_Lab2BGR)
    
    # Calculate final mean luminance and contrast
    final_mean, final_stdev = calculate_luminance_contrast(normalized_rgb)
    print(f"Final luminance: {final_mean}, Final contrast: {final_stdev}\n")
    
    # Save the image
    filename = os.path.basename(image_path)
    output_path = os.path.join(output_directory, filename)
    cv2.imwrite(output_path, normalized_rgb)

def process_directory(input_dir, output_dir, target_luminance, target_contrast):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    image_paths = []
    for root, _, files in os.walk(input_dir):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                image_paths.append(os.path.join(root, file))

    if not image_paths:
        print("No images found in the input directory.")
        return

    print(f"Target luminance: {target_luminance}, Target contrast: {target_contrast}")

    for image_path in image_paths:
        normalize_luminance(image_path, output_dir, target_luminance, target_contrast)

    print("Processing complete.")

def verify_luminance_contrast_lab(output_dir, target_luminance, target_contrast, tolerance=3):
    luminance_list = []
    contrast_list = []

    for root, _, files in os.walk(output_dir):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                image_path = os.path.join(root, file)
                print(f"Verifying image: {image_path}")
                lab_image = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2Lab)
                luminance, contrast = calculate_luminance_contrast(lab_image)
                luminance_list.append(luminance)
                contrast_list.append(contrast)
                lum_error = luminance - target_luminance
                contrast_error = contrast - target_contrast
                print(f"Luminance error: {lum_error}, Contrast error: {contrast_error}")

    mean_luminance_error = np.mean([l - target_luminance for l in luminance_list])
    mean_contrast_error = np.mean([c - target_contrast for c in contrast_list])

    print(f"Mean luminance error: {mean_luminance_error}, Mean contrast error: {mean_contrast_error}")
    print("Verification complete.")
