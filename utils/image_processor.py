import time

import numpy as np
from scipy import ndimage
import cv2 as cv

from skimage.segmentation import watershed
from skimage.feature import peak_local_max


def apply_filters(img, gaussian=False, grayscale=True):
    filtered_image = img

    # In case the values we have from the image are [0-255] we normalize them
    if filtered_image.max() > 1:
        filtered_image = filtered_image/255

    if grayscale:
        filtered_image = grayscale_filter(filtered_image)
    if gaussian:
        filtered_image = ndimage.gaussian_filter(filtered_image, sigma=.1)

    return filtered_image


def grayscale_filter(img):
    return np.dot(img[..., :3], [0.2989, 0.5870, 0.1140])


# Will return the color of the circle based on the intensity of the pixel
def get_outline_color(pixel_value):
    if pixel_value > .8:
        return '#660000'
    elif pixel_value > .7:
        return '#ff1919'
    elif pixel_value > .6:
        return '#ff6666'
    elif pixel_value > .5:
        return '#ff8000'
    else:
        return '#FFFF00'


def sobel(img):
    # ret, binary = cv.threshold(img, 0, 255, cv.THRESH_OTSU | cv.THRESH_BINARY_INV)

    # sobel = cv.Sobel(img, cv.CV_64F, 0, 1, ksize=9)
    before = time.time()
    grad_x = cv.Sobel(img, cv.CV_64F, 1, 0, ksize=3)
    grad_y = cv.Sobel(img, cv.CV_64F, 0, 1, ksize=3)

    abs_grad_x = cv.convertScaleAbs(grad_x)
    abs_grad_y = cv.convertScaleAbs(grad_y)

    result = cv.addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0)
    after = time.time()

    return result, round(after-before, 5)
    # return get_circles(cv.addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0))


def laplacian(img):
    before = time.time()

    # return cv.Laplacian(np.float32(img), cv.CV_16S, ksize=3)
    # grayimg = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    laplacian = cv.Laplacian(img, cv.CV_8U, ksize=3)
    # ret, thresh = cv.threshold(laplacian, .5, 1, cv.THRESH_BINARY)
    after = time.time()

    return laplacian, round(after-before, 5)
    # return get_circles(laplacian)
    # return cv.Canny(image=np.uint8(laplacian), threshold1=100, threshold2=200)  # Canny Edge Detection


def thresholding(img):
    before = time.time()
    _, binary = cv.threshold(img, np.mean(img), 255, cv.THRESH_BINARY)
    after = time.time()
    return binary, round(after-before, 5)
    # return get_circles(binary)


def canny(img):
    before = time.time()
    result = cv.Canny(image=img, threshold1=100, threshold2=200)
    after = time.time()
    # return get_circles(cv.Canny(image=img, threshold1=100, threshold2=200))
    return result, round(after-before, 5)  # Canny Edge Detection


def watersheding(img):
    before = time.time()
    distance = ndimage.distance_transform_edt(img)
    coords = peak_local_max(distance, footprint=np.ones((3, 3)), labels=img)
    mask = np.zeros(distance.shape, dtype=bool)
    mask[tuple(coords.T)] = True
    markers, _ = ndimage.label(mask)
    labels = watershed(-distance, markers, mask=img)
    after = time.time()
    return labels, round(after-before, 5)
    # return get_circles(labels)


def harris(img):
    before = time.time()
    dst = cv.cornerHarris(np.float32(img), 2, 3, 0.04)
    img[dst > 0.01 * dst.max()] = 1
    return img, round(time.time()-before, 5)


def get_circles(img):
    hh, ww = img.shape[:2]
    min_dist = int(ww/10)
    circles = cv.HoughCircles(img, cv.HOUGH_GRADIENT, 1, minDist=min_dist, param1=50, param2=10, minRadius=0,
                               maxRadius=0)
    # draw circles
    result = img.copy()
    for circle in circles[0]:
        # draw the circle in the output image
        (x, y, r) = circle
        x = int(x)
        y = int(y)
        r = int(r/4)
        cv.circle(result, (x, y), r, (255, 255, 255), 1)
    return result
