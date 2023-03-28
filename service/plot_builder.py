import matplotlib.pyplot as plt
from pathlib import Path
import matplotlib.image as mpimg
from segmentation import custom_processor as cp
from utils import image_processor as ip
import numpy as np
import cv2 as cv


def display_images(identified_objects, catalog_images_path):
    for identified_object in identified_objects:
        segment = identified_object['segment']
        path = Path(__file__).parent / catalog_images_path / identified_object['filename']
        catalog_image = mpimg.imread(path)

        f, axarr = plt.subplots(1, 2)
        plt.suptitle('Segment ' + str(identified_object['segment']) + '-' + identified_object['filename'] +
                     ' (Error value ' + str(round(identified_object['err'], 5)) + ')')
        axarr[0].imshow(cp.astronomical_objects[segment-1], cmap='gray')

        catalog_image = ip.apply_filters(catalog_image)
        axarr[1].imshow(catalog_image, cmap='gray')

        plt.show()


def display_images_one_window(identified_objects, catalog_images_path):
    row_index = 0
    col_index = 0
    split_point = int(np.floor(np.sqrt(len(identified_objects))))
    row_reset_count = 0
    # Display the comparison between images
    f, axarr = plt.subplots(split_point+1, split_point*2)
    plt.tight_layout()
    for identified_object in identified_objects:
        segment = identified_object['segment']
        path = Path(__file__).parent / catalog_images_path / identified_object['filename']
        catalog_image = mpimg.imread(path)

        # f, axarr = plt.subplots(1, 2)
        plt.suptitle('Segment ' + str(identified_object['segment']) + '-' + identified_object['filename'] +
                     ' (Error value ' + str(identified_object['err']) + ')')
        axarr[row_index, col_index].imshow(ip.astronomical_objects[segment], cmap='gray')

        catalog_image = ip.apply_filters(catalog_image)
        axarr[row_index, col_index+1].imshow(catalog_image, cmap='gray')

        row_reset_count += 1

        col_index += 2
        if row_reset_count >= split_point:
            row_index += 1
            col_index = 0
            row_reset_count = 0

    plt.show()


def display_two_images(imageA, imageB, title=''):
    f, axarr = plt.subplots(1, 2)
    plt.suptitle(title)
    axarr[0].imshow(imageA, cmap='gray')
    axarr[1].imshow(imageB, cmap='gray')
    plt.show()


def display_image(image, title=''):
    # get_circles(image)
    f, axarr = plt.subplots(1)
    plt.suptitle(title)
    axarr.imshow(image, cmap='gray')
    plt.show()


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