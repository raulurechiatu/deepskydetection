import numpy as np
from matplotlib import pyplot as plt

import service.image_comparator as ic
import service.image_loader as il
import utils.image_processor as ip
import service.db_manager as db
import service.plot_builder as plot
import service.train_service as ts
import service.data_service as ds
from segmentation import custom_processor as cp
import cv2


ts.configure_gpu()


# Path for the initial image to start the algorithm on
# This image is for example a picture took with a personal telescope or obtain via the internet to be analyzed
images_parent_path = "../images/deepsky/"
image_name = 'andromeda'

galaxy_zoo_images_path = "../resources/galaxyzoo2/images_gz2/images/"
nebulae_images_path = "../resources/nebulae/images/"
galaxies_images_path = "../resources/galaxies/"
stars_images_path = "../resources/stars/"

# Worked with 10k
images_to_load = 4000
rotations = 12

error_threshold = 0.85
# error_threshold = 0.0016
# error_threshold = 0.04
total_threads_number = 5


def get_class_name(class_number):
    if class_number == 0:
        return "Round smooth"
    if class_number == 1:
        return "In-between smooth"
    if class_number == 2:
        return "Cigar-shaped smooth"
    if class_number == 3:
        return "Edge-on"
    if class_number == 4:
        return "Spiral"


def compare_data():
    identified_objects = ic.start_comparison_process(images_parent_path + str(image_name + '.png'),
                                                     galaxy_zoo_images_path,
                                                     total_threads_number, error_threshold,
                                                     images_to_load=images_to_load,
                                                     zoom_from_center=25)

    if len(identified_objects) == 0:
        print("No similarities found on the catalog")
        exit()

    # Load the db files and search for a filename
    # db.load_dbs()
    # db.search_file(identified_objects)

    plot.display_images(identified_objects, galaxy_zoo_images_path)

    ic.download_segmented_objects(image_name)
    # il.load_images(galaxy_zoo_images_path, 2000, 0)

    # il.compare_filters("test")


def compare_segmentation():
    # algorithms = ['custom', 'sobel', 'laplace', 'threshold']
    algorithm = 'custom'
    # for algorithm in algorithms:
    _, exec_time = il.compare_segmentation_algorithms(images_parent_path + str(image_name + '.png'), image_name, download_segmented=False, display_images=True, algorithm=algorithm)
    if exec_time != -1:
        print("Execution time for algorithm ", algorithm, " is ", exec_time, "s")
    else:
        print("For the execution time please call the method with the value of the display_images parameter set to True!")


def train_data():
    galaxy_images, galaxy_image_names = il.load_images(galaxy_zoo_images_path, images_to_load, 0, random=False)
    # Load the db files and search for a filename
    galaxy_data = db.get_data(galaxy_image_names)

    _, indexed_labels = db.get_labels(galaxy_data)
    galaxy_images, indexed_labels = ds.remove_class(galaxy_images, indexed_labels, 5)

    # Get images rotated by the parameter number of times and the labels multiplied by the same number
    galaxy_images, indexed_labels = il.get_rotations(galaxy_images, indexed_labels, rotations)
    galaxy_images = galaxy_images / 255.0

    ts.train(galaxy_images, indexed_labels, galaxy_image_names)


def evaluate_image(model_name=None):
    original_image = il.load_image_matplot(images_parent_path + str(image_name + '.png'))
    original_image = ip.apply_filters(original_image, gaussian=False)
    cp.identify_and_outline_objects(original_image, outline=False, save=True, zoom_from_center=15)
    images = np.array(cp.astronomical_objects)
    final_images = []
    for i in range(len(images)):
        final_images.append(il.resize(images[i]))

    final_images = np.array(final_images)
    final_images = final_images.reshape(-1, 1, ts.number_of_pixels, ts.number_of_pixels)
    ts.evaluate(final_images, None, model_name=model_name)

    f, axarr = plt.subplots(1)
    axarr.imshow(original_image, cmap='gray')
    cp.identify_and_outline_objects(original_image, plt=plt, outline=True, save=True)
    plt.show()


def evaluate_data(evaluation_images_number, model_name=None):
    galaxy_images, galaxy_image_names = il.load_images(galaxy_zoo_images_path, evaluation_images_number, 0, random=False)
    galaxy_images_r, galaxy_image_names_r = il.load_images(galaxy_zoo_images_path, round(evaluation_images_number/5), 0, random=True)
    galaxy_images = np.concatenate((galaxy_images, galaxy_images_r))
    galaxy_image_names = np.concatenate((galaxy_image_names, galaxy_image_names_r))
    galaxy_data = db.get_data(galaxy_image_names)

    _, indexed_labels = db.get_labels(galaxy_data)
    galaxy_images, indexed_labels = ds.remove_class(galaxy_images, indexed_labels, 5)
    galaxy_images = galaxy_images / 255.0
    results = ts.evaluate(galaxy_images, indexed_labels, model_name, manual=True)
    print("results(predicted, actual): ", results)
    for i in range(len(results)):
        title = "Predicted: " + str(results[i][0]) + " (" + get_class_name(results[i][0]) + ")  |  " + " Actual: " + str(results[i][1]) + " (" + get_class_name(results[i][1]) + ")"
        plot.display_image(galaxy_images[i], title)


def live_detection():
    # cv2.namedWindow("preview")
    vc = cv2.VideoCapture(0)
    if vc.isOpened():  # try to get the first frame
        rval, frame = vc.read()
    else:
        rval = False

    while rval:
        rval, frame = vc.read()
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # original_image = ip.apply_filters(frame, gaussian=False)
        f, axarr = plt.subplots(1)
        axarr.imshow(gray_frame, cmap='gray')
        cp.identify_and_outline_objects(gray_frame, plt=plt, outline=True, save=True)
        plt.show()
        # cv2.imshow("preview", grayFrame)

        key = cv2.waitKey(20)
        if key == 27:  # exit on ESC
            break

    vc.release()
    # cv2.destroyWindow("preview")


if __name__ == '__main__':
    train_data()
    # evaluate_image()
    # evaluate_data(3000, "valid/L_CUSTOM_2_3_64_90240_10ep_96.37acc.h5")
    # live_detection()

    # print(multiprocessing.cpu_count())
    # compare_data()
    # compare_segmentation()
