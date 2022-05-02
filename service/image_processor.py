
import numpy as np
from scipy.ndimage import gaussian_filter


def process(img, gaussian=False, grayscale=True):
    filtered_image = img

    if grayscale:
        filtered_image = grayscale_filter(img)
    if gaussian:
        filtered_image = gaussian_filter(filtered_image, sigma=1)

    return filtered_image


def grayscale_filter(img):
    return np.dot(img[...,:3], [0.2989, 0.5870, 0.1140])


def identify_object(img):
    coords = [100, 100]
    return coords