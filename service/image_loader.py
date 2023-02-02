import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from PIL import Image
from utils import progressbar, image_processor as ip
from segmentation import custom_processor as cp
import cv2
import numpy as np

import shutil
import os
import time
from pathlib import Path
from service.train_service import number_of_pixels

segmentation_path = "images/output/segmentation/"

# galaxyzoo_images = []


def load_images(folder_path, images_to_load=-1, offset=0):
    # global galaxyzoo_images
    galaxyzoo_images = np.empty(shape=(images_to_load, number_of_pixels, number_of_pixels), dtype=np.int8)
    final_path = Path(__file__).parent / folder_path

    before = time.time()
    image_names = os.listdir(final_path)
    after = time.time()
    print("Execution time for listdir(getting the images name) is ", (after - before), "s for ", len(image_names),
          " images")
    image_number = offset

    # The reason behind this if is to iterate more efficiently over the whole dataset without adding a check step
    # each time
    if images_to_load == -1:
        progressbar.printProgressBar(0, len(image_names), prefix='Progress:', suffix='Complete', length=50)
        for image_name in image_names:
            progressbar.printProgressBar(image_names.index(image_name), len(image_names),
                                         prefix='Loading images in memory:', suffix='Complete', length=50)
            # Here we can change the library used to load images in memory
            # galaxyzoo_images.append(load_image_cv(folder_path + image_name))
            # np.add(galaxyzoo_images, load_image_cv(folder_path + image_name))
            galaxyzoo_images[image_names.index(image_name)] = load_image_cv(folder_path + image_name)

    else:
        progressbar.printProgressBar(0, images_to_load, prefix='Progress:', suffix='Complete', length=50)
        for image_name in image_names:
            image_number = image_number + 1
            progressbar.printProgressBar(image_number, images_to_load, prefix='Loading images in memory:',
                                         suffix='Complete', length=50)
            # Here we can change the library used to load images in memory
            # galaxyzoo_images.append(load_image_cv(folder_path + image_name))
            # np.add(galaxyzoo_images, load_image_cv(folder_path + image_name))
            galaxyzoo_images[image_names.index(image_name)] = load_image_cv(folder_path + image_name)
            if images_to_load == image_number:
                break

    after_image_load = time.time()
    print("Execution time for loading images in memory is ", (after_image_load - after), "seconds for ", images_to_load,
          " images")
    # print(galaxyzoo_images)

    return galaxyzoo_images, image_names


def compare_segmentation_algorithms(path, image_name, download_segmented=False, display_images=False,
                                    algorithm='custom'):
    # Load the image
    original = load_image_matplot(path)
    original_cv = load_image_cv(path, 0)
    exec_time = -1

    if display_images:
        # Process the image
        # original = ip.apply_filters(original, gaussian=True)
        image_custom = original.copy()
        filtered = ip.apply_filters(image_custom)
        # filtered_copy = filtered.copy()
        f, axarr = plt.subplots(3, 3)

        axarr[0][0].imshow(original, cmap='gray')
        axarr[0][0].set_title('Original')
        # axarr[1][0].imshow(original_cv, cmap='gray')
        axarr[0][1].imshow(filtered, cmap='gray')
        axarr[0][1].set_title('Filtered')
        harris, harris_time = ip.harris(original_cv)
        axarr[0][2].imshow(harris, cmap='gray')
        axarr[0][2].set_title('Harris (' + str(harris_time) + 's)')

        laplace, laplace_time = ip.laplacian(original_cv)
        axarr[1][0].imshow(laplace, cmap='gray')
        axarr[1][0].set_title('Laplacian (' + str(laplace_time) + 's)')
        sobel, sobel_time = ip.sobel(original_cv)
        axarr[1][1].imshow(sobel, cmap='gray')
        axarr[1][1].set_title('Sobel (' + str(sobel_time) + 's)')
        threshold, threshold_time = ip.thresholding(original_cv)
        axarr[1][2].imshow(threshold, cmap='gray')
        axarr[1][2].set_title('Thresholding (' + str(threshold_time) + 's)')

        watershed, watershed_time = ip.watersheding(original_cv)
        axarr[2][0].imshow(watershed, cmap='gray')
        axarr[2][0].set_title('Watershed (' + str(watershed_time) + 's)')
        canny, canny_time = ip.canny(original_cv)
        axarr[2][1].imshow(canny, cmap='gray')
        axarr[2][1].set_title('Canny (' + str(canny_time) + 's)')
        cp.identify_and_outline_objects(filtered, axarr[2][2])
        axarr[2][2].imshow(filtered, cmap='gray')
        # axarr[0][2].imshow(ip.get_circles(np.uint8(filtered*255)), cmap='gray')
        axarr[2][2].set_title('Proposed (' + str(round(cp.exec_time, 5)) + 's)')

        # axarr[2][2].imshow(original, cmap='gray')
        # axarr[2][2].set_title('Original')

        plt.show()

    # Will download the images under segmentation_path variable value if the option is selected
    # The method will generate the astronomical objects segmented out of the image
    if download_segmented:
        download_segmented_objects(image_name)

    return original, exec_time


# 10k images: 24.8s
# 1k images: 2.3s
# Used for an efficient way to load the images
def load_image_matplot(path):
    # Load the image
    final_path = Path(__file__).parent / path
    original = mpimg.imread(final_path)
    # if grayscale:
    original = ip.apply_filters(original)
    return original


# 1k images: 2.3s
def load_image_pil(path):
    final_path = Path(__file__).parent / path
    original = Image.open(final_path)
    # grayscale
    original = original.convert('L')
    return original.getdata()


def load_image_cv(path, grayscale=0, resize=1):
    final_path = Path(__file__).parent / path
    original = cv2.imread(str(final_path), grayscale)
    desired_shape = (number_of_pixels, number_of_pixels)
    if original is None:
        return np.zeros(desired_shape)
    if original.shape != desired_shape:
        original = cv2.resize(original, desired_shape, interpolation=cv2.INTER_AREA)
    return original


# Deprecated DO NOT USE
def compare_filters(path):
    # Load the image
    path = Path(__file__).parent / path
    image_mp = mpimg.imread(path)
    image_mp2 = mpimg.imread(path)
    # process_images(image_mp)

    # Process the image
    image_mp = cp.process(image_mp, gaussian=False)
    image_mp2 = cp.process(image_mp2, gaussian=True)

    # Display the comparison between images
    f, axarr = plt.subplots(1, 2)
    axarr[0].imshow(image_mp, cmap='gray')
    axarr[1].imshow(image_mp2, cmap='gray')
    plt.show()


# Will download the images saved by the image_processor
# The name will be generated based on the image name, an index and the size of the kernel
# The size of the kernel is the diagonal of the matrix generated from the identified center of the image
def download_segmented_objects(image_name):
    if len(cp.astronomical_objects) > 0:
        path = Path(__file__).parent.parent / segmentation_path / image_name
        if path.exists():
            shutil.rmtree(path)
        path.mkdir(exist_ok=True)
    else:
        print("No objects identified on the image")
        return

    index = 0
    for image in cp.astronomical_objects:
        im_name = segmentation_path + image_name + "/" + str(index) + '_' + str(len(image)) + ".png"
        mpimg.imsave(im_name, image, cmap='gray')
        index += 1
