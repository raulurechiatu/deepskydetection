import matplotlib.pyplot as plt
from pathlib import Path
import matplotlib.image as mpimg
from service import image_processor as ip
import numpy as np


def display_images(identified_objects, catalog_images_path):
    for identified_object in identified_objects:
        segment = identified_object['segment']
        path = Path(__file__).parent / catalog_images_path / identified_object['filename']
        catalog_image = mpimg.imread(path)

        f, axarr = plt.subplots(1, 2)
        plt.suptitle('Segment ' + str(identified_object['segment']) + '-' + identified_object['filename'] +
                     ' (Error value ' + str(round(identified_object['err'], 5)) + ')')
        axarr[0].imshow(ip.astronomical_objects[segment], cmap='gray')

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


def display_two_images(imageA, imageB):
    f, axarr = plt.subplots(1, 2)
    axarr[0].imshow(imageA, cmap='gray')
    axarr[1].imshow(imageB, cmap='gray')
    plt.show()