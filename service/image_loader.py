
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from PIL import Image
from main import image_name

import shutil
import os
import time
from pathlib import Path
from service import image_processor


segmentation_path = "images/output/segmentation/"
# galaxyzoo_images = []


def load_images(folder_path, images_to_load=-1, offset=0):
    # global galaxyzoo_images
    # galaxyzoo_images = np.ndarray(shape=(424, 424, 3))
    galaxyzoo_images = []
    final_path = Path(__file__).parent / folder_path

    before = time.time()
    image_names = os.listdir(final_path)
    after = time.time()
    print("Execution time for listdir(getting the images name) is ", (after-before), "s for ", len(image_names), " images")
    image_number = offset

    # The reason behind this if is to iterate more efficiently over the whole dataset without adding a check step
    # each time
    if images_to_load == -1:
        for image_name in image_names:
            # Here we can change the library used to load images in memory
            galaxyzoo_images.append(load_image_matplot(folder_path + image_name))

    else:
        for image_name in image_names:
            image_number = image_number + 1
            # Here we can change the library used to load images in memory
            galaxyzoo_images.append(load_image_matplot(folder_path + image_name))
            if images_to_load == image_number:
                break

    after_image_load = time.time()
    print("Execution time for loading images in memory is ", (after_image_load-after), "seconds for ", images_to_load, " images")
    # print(galaxyzoo_images)

    return galaxyzoo_images, image_names


# Loads one image and has several options in order to download segmented images out of that or to display that image
# This is meant for better manipulation of the data and powerful observation tools while the other function should be
# used for bulk loading of images
# Deprecated DO NOT USE
def load_image_prettified(path, download_segmented=False, display_images=False):
    # Load the image
    original = load_image_matplot(path)

    if display_images:
        # Process the image
        original = image_processor.apply_filters(original, gaussian=True)
        image_mp = original
        f, axarr = plt.subplots(1, 2)

        # Display the image
        image_processor.identify_and_outline_objects(image_mp, axarr[1])
        axarr[0].imshow(original, cmap='gray')
        axarr[1].imshow(image_mp, cmap='gray')
        # plt.imshow(image_mp, cmap='gray')
        plt.show()

    # Will download the images under segmentation_path variable value if the option is selected
    # The method will generate the astronomical objects segmented out of the image
    if download_segmented:
        download_segmented_objects()

    return original


# 10k images: 24.8s
# 1k images: 2.3s
# Used for an efficient way to load the images
def load_image_matplot(path):
    # Load the image
    final_path = Path(__file__).parent / path
    original = mpimg.imread(final_path)
    # if grayscale:
    #     original = image_processor.apply_filters(original)
    return original


# 1k images: 2.3s
def load_image_pil(path):
    final_path = Path(__file__).parent / path
    original = Image.open(final_path)
    # grayscale
    # original = original.convert('L')
    return list(original.getdata())


# Deprecated DO NOT USE
def compare_filters(path):
    # Load the image
    path = Path(__file__).parent / path
    image_mp = mpimg.imread(path)
    image_mp2 = mpimg.imread(path)
    # process_images(image_mp)

    # Process the image
    image_mp = image_processor.process(image_mp, gaussian=False)
    image_mp2 = image_processor.process(image_mp2, gaussian=True)

    # Display the comparison between images
    f, axarr = plt.subplots(1, 2)
    axarr[0].imshow(image_mp, cmap='gray')
    axarr[1].imshow(image_mp2, cmap='gray')
    plt.show()


# Will download the images saved by the image_processor
# The name will be generated based on the image name, an index and the size of the kernel
# The size of the kernel is the diagonal of the matrix generated from the identified center of the image
def download_segmented_objects():
    if len(image_processor.astronomical_objects) > 0:
        path = Path(__file__).parent.parent / segmentation_path / image_name
        if path.exists():
            shutil.rmtree(path)
        path.mkdir(exist_ok=True)
    else:
        print("No objects identified on the image")
        return

    index = 0
    for image in image_processor.astronomical_objects:
        im_name = segmentation_path + image_name + "/" + str(index) + '_' + str(len(image)) + ".png"
        mpimg.imsave(im_name, image, cmap='gray')
        index += 1
