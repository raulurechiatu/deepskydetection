import service.image_loader as il
# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

# Path for the initial image to start the algorithm on
# This image is for example a picture took with a personal telescope or obtain via the internet to be analyzed
images_parent_path = "../images/deepsky/"
image_name = 'andromeda'

galaxy_zoo_images_path = "../resources/galaxyzoo2/images_gz2/images/"


def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    il.load_images(galaxy_zoo_images_path, 2000, 0)
    # il.load_image_prettified(images_parent_path + str(image_name + '.png'), download_segmented=False, display_images=True)

    # il.compare_filters("test")

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
