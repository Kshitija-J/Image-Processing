import numpy as np
import imageio
import os

def histogram_equalization(image):
    hist, bins = np.histogram(image.flatten(), 256, [0, 256])
    cdf = hist.cumsum()
    cdf_normalized = (cdf - cdf[0]) * 255 / (cdf[-1] - cdf[0])
    cdf_normalized = cdf_normalized.astype(np.uint8)
    equalized_img = cdf_normalized[image]
    return equalized_img

def log_transformation(image, c=1):
    image_float = np.where(image > 0, image.astype('float32'), 1)  # Avoid log(0)
    output = c * np.log(1 + image_float)
    output_normalized = (output - output.min()) / (output.max() - output.min()) * 255
    return output_normalized.astype(np.uint8)



def rotate_image(image, angle):
    angle_rad = np.radians(angle)
    cos, sin = np.cos(angle_rad), np.sin(angle_rad)
    
    # Adjust for color/grayscale images
    if image.ndim == 3:
        n, m, _ = image.shape
    else:
        n, m = image.shape

    center = np.array([m / 2, n / 2])

    rotated_img = np.zeros_like(image)

    for i in range(n):
        for j in range(m):
            offset = np.array([j, i]) - center
            original_j, original_i = center + np.array([cos*offset[0] - sin*offset[1], sin*offset[0] + cos*offset[1]])
            original_i, original_j = int(original_i), int(original_j)
            if 0 <= original_i < n and 0 <= original_j < m:
                if image.ndim == 3:  # For color images
                    rotated_img[i, j, :] = image[original_i, original_j, :]
                else:  # For grayscale images
                    rotated_img[i, j] = image[original_i, original_j]

    return rotated_img





def gaussian_kernel(size, sigma=1):
    size = int(size) // 2
    x, y = np.mgrid[-size:size+1, -size:size+1]
    g = np.exp(-(x**2 + y**2) / (2*sigma**2))
    kernel = g / g.sum()
    return kernel

def gaussian_blur(image, kernel_size=3, sigma=1):
    kernel = gaussian_kernel(kernel_size, sigma)
    pad_width = kernel_size // 2
    if image.ndim == 3:  # Color image
        h, w, channels = image.shape
        padded_image = np.pad(image, ((pad_width, pad_width), (pad_width, pad_width), (0, 0)), mode='constant', constant_values=0)
        blurred = np.zeros_like(image)
        for c in range(channels):
            for i in range(h):
                for j in range(w):
                    blurred[i, j, c] = (kernel * padded_image[i:i+kernel_size, j:j+kernel_size, c]).sum()
    else:  # Grayscale image
        h, w = image.shape
        padded_image = np.pad(image, pad_width, mode='constant', constant_values=0)
        blurred = np.zeros_like(image)
        for i in range(h):
            for j in range(w):
                blurred[i, j] = (kernel * padded_image[i:i+kernel_size, j:j+kernel_size]).sum()
    return blurred


def median_filter(image, kernel_size=3):
    pad_width = kernel_size // 2
    padded_image = np.pad(image, pad_width, mode='constant', constant_values=0)
    filtered = np.zeros_like(image)
    
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            filtered[i, j] = np.median(padded_image[i:i+kernel_size, j:j+kernel_size])
    return filtered

def load_image(file_path):
    return imageio.imread(file_path)

def save_image(image, file_path):
    imageio.imwrite(file_path, image)

def process_images(directory, image_files):
    for image_file in image_files:
        full_path = os.path.join(directory, image_file)
        print(f"Processing {full_path}...")
        
        image = load_image(full_path)
        
        if image is not None:
            equalized_image = histogram_equalization(image)
            log_image = log_transformation(image)
            rotated_image = rotate_image(image, 45)  
            blurred_image = gaussian_blur(image)
            median_image = median_filter(image)
            
            save_image(equalized_image, os.path.join(directory, f'{image_file}_equalized.png'))
            save_image(log_image, os.path.join(directory, f'{image_file}_log.png'))
            save_image(rotated_image, os.path.join(directory, f'{image_file}_rotated.png'))
            save_image(blurred_image, os.path.join(directory, f'{image_file}_blurred.png'))
            save_image(median_image, os.path.join(directory, f'{image_file}_median.png'))
        else:
            print(f"Failed to load {image_file}. Skipping...")

images_directory = 'C:\\Users\\kshit\\.vscode\\Project_nolibraries'
image_files = ["auto.pnm", "building.pnm", "child.pnm", "ct_scan.pnm", "tire.pnm"]

process_images(images_directory, image_files)
