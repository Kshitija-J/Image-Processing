import cv2
import numpy as np
import os

def histogram_equalization_opencv(image):
    if len(image.shape) == 2:  # Grayscale image
        return cv2.equalizeHist(image)
    else:  
        image_yuv = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
        image_yuv[:, :, 0] = cv2.equalizeHist(image_yuv[:, :, 0])
        return cv2.cvtColor(image_yuv, cv2.COLOR_YUV2BGR)

def gaussian_blur_opencv(image, kernel_size=3, sigma=1):
    return cv2.GaussianBlur(image, (kernel_size, kernel_size), sigma)

def median_filter_opencv(image, kernel_size=3):
    return cv2.medianBlur(image, kernel_size)

def process_images_opencv(directory, image_files):
    for image_file in image_files:
        full_path = os.path.join(directory, image_file)
        print(f"Processing with OpenCV: {full_path}")
        
        image = cv2.imread(full_path, cv2.IMREAD_UNCHANGED)
        
        if image is not None:
            equalized_image = histogram_equalization_opencv(image)
            blurred_image = gaussian_blur_opencv(image)
            median_image = median_filter_opencv(image)
            
            # Save processed images
            cv2.imwrite(os.path.join(directory, f'{image_file}_opencv_equalized.png'), equalized_image)
            cv2.imwrite(os.path.join(directory, f'{image_file}_opencv_blurred.png'), blurred_image)
            cv2.imwrite(os.path.join(directory, f'{image_file}_opencv_median.png'), median_image)
        else:
            print(f"Failed to load {image_file} with OpenCV. Skipping...")

images_directory = 'C:\\Users\\kshit\\.vscode\\Project_nolibraries'
image_files = ["auto.pnm", "building.pnm", "child.pnm", "ct_scan.pnm", "tire.pnm"]

process_images_opencv(images_directory, image_files)
