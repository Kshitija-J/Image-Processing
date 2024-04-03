import numpy as np
import imageio
import matplotlib.pyplot as plt
import os

def gaussian_kernel(size, sigma=1):
    size = int(size) // 2
    x, y = np.mgrid[-size:size+1, -size:size+1]
    g = np.exp(-(x**2 + y**2) / (2 * sigma**2))
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
    return blurred.astype('uint8')

def median_filter(image, kernel_size=3):
    pad_width = kernel_size // 2
    padded_image = np.pad(image, pad_width, mode='constant', constant_values=0)
    filtered = np.zeros_like(image)
    
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            if image.ndim == 3:  # Color image
                for c in range(image.shape[2]):
                    neighborhood = padded_image[i:i+kernel_size, j:j+kernel_size, c]
                    filtered[i, j, c] = np.median(neighborhood)
            else:  # Grayscale image
                neighborhood = padded_image[i:i+kernel_size, j:j+kernel_size]
                filtered[i, j] = np.median(neighborhood)
    return filtered.astype('uint8')



def add_salt_pepper_noise(image, salt_prob=0.005, pepper_prob=0.005):
    noisy_image = np.copy(image)
    # Salt noise
    num_salt = np.ceil(salt_prob * image.size)
    coords_salt = (np.random.randint(0, i - 1, int(num_salt)) for i in image.shape[:2])
    noisy_image[tuple(coords_salt)] = 255
    
    # Pepper noise
    num_pepper = np.ceil(pepper_prob * image.size)
    coords_pepper = (np.random.randint(0, i - 1, int(num_pepper)) for i in image.shape[:2])
    noisy_image[tuple(coords_pepper)] = 0

    return noisy_image.astype('uint8')

def show_images(original, noisy, gaussian, median, title='Image', cmap=None):
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    axes[0].imshow(original, cmap=cmap)
    axes[0].set_title('Original ' + title)
    axes[0].axis('off')
    
    axes[1].imshow(noisy, cmap=cmap)
    axes[1].set_title('Salt & Pepper Noise')
    axes[1].axis('off')
    
    axes[2].imshow(gaussian, cmap=cmap)
    axes[2].set_title('Gaussian Blur')
    axes[2].axis('off')
    
    axes[3].imshow(median, cmap=cmap)
    axes[3].set_title('Median Filter')
    axes[3].axis('off')
    
    plt.show()

def process_and_show_images(directory, image_files):
    for image_file in image_files:
        full_path = os.path.join(directory, image_file)
        print(f"Processing {full_path}...")
        image = imageio.imread(full_path)
        
        # Check if the image is grayscale and set the cmap accordingly
        cmap = 'gray' if image.ndim == 2 else None
        
        noisy_image = add_salt_pepper_noise(image, salt_prob=0.01, pepper_prob=0.01)
        gaussian_filtered_image = gaussian_blur(noisy_image, kernel_size=3, sigma=1)
        median_filtered_image = median_filter(noisy_image, kernel_size=3)
        
        show_images(image, noisy_image, gaussian_filtered_image, median_filtered_image, title=image_file, cmap=cmap)

images_directory = 'C:\\Users\\kshit\\.vscode\\Project_nolibraries'
image_files = ["auto.pnm", "building.pnm", "child.pnm", "ct_scan.pnm", "tire.pnm"]

process_and_show_images(images_directory, image_files)
