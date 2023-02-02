import service.image_comparator as ic
import service.image_loader as il
import service.db_manager as db
import service.plot_builder as plot
import service.train_service as ts


ts.configure_gpu()

# Path for the initial image to start the algorithm on
# This image is for example a picture took with a personal telescope or obtain via the internet to be analyzed
images_parent_path = "../images/deepsky/"
image_name = 'andromeda'

galaxy_zoo_images_path = "../resources/galaxyzoo2/images_gz2/images/"
nebulae_images_path = "../resources/nebulae/images/"
galaxies_images_path = "../resources/galaxies/"
stars_images_path = "../resources/stars/"

images_to_load = 2000
error_threshold = 0.85
# error_threshold = 0.0016
# error_threshold = 0.04
total_threads_number = 5


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
    galaxy_images, _ = il.load_images(galaxy_zoo_images_path, images_to_load, 0)
    nebulae_images, _ = il.load_images(nebulae_images_path, images_to_load, 0)
    star_images, _ = il.load_images(stars_images_path, images_to_load, 0)
    ts.train(galaxy_images, nebulae_images, star_images)


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    train_data()
    # print(multiprocessing.cpu_count())
    # compare_data()
    # compare_segmentation()
