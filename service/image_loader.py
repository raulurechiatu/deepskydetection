
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from pathlib import Path
import shutil
from service import image_processor


images_parent_path = "../images/deepsky/"
segmentation_path = "images/output/segmentation/"
image_name = 'alfa_centauri'


def load_images(path, download_segmented=False):
    # Load the image
    path = Path(__file__).parent / images_parent_path / str(image_name + '.png')
    original = mpimg.imread(path)

    # Process the image
    original = image_processor.apply_filters(original, gaussian=True)
    image_mp = original
    f, axarr = plt.subplots(1, 2)

    # Display the image
    image_processor.identify_and_outline_objects(axarr[1], image_mp)

    # Will download the images under segmentation_path variable value if the option is selected
    # The method will generate the astronomical objects segmented out of the image
    if download_segmented:
        download_segmented_objects()

    axarr[0].imshow(original, cmap='gray')
    axarr[1].imshow(image_mp, cmap='gray')
    # plt.imshow(image_mp, cmap='gray')
    plt.show()


def compare_filters(path):
    # Load the image
    path = Path(__file__).parent / images_parent_path / 'andromeda.png'
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
