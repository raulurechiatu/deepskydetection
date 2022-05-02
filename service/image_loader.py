
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from pathlib import Path
from service import image_processor

images_parent_path = "images/"


def load_images(path):
    # Load the image
    path = Path(__file__).parent / '../images/deepsky/andromeda.png'
    image_mp = mpimg.imread(path)
    # process_images(image_mp)

    # Process the image
    image_mp = image_processor.process(image_mp, gaussian=False)

    # Display the image
    plt.imshow(image_mp, cmap='gray')
    plt.show()


def compare_filters(path):
    # Load the image
    path =  Path(__file__).parent / '../images/deepsky/andromeda.png'
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


def process_images(image_mp):
    print()



