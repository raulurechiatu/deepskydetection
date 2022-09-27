import service.image_comparator as ic
import service.db_manager as db
import service.plot_builder as plot

# Path for the initial image to start the algorithm on
# This image is for example a picture took with a personal telescope or obtain via the internet to be analyzed
images_parent_path = "../images/deepsky/"
image_name = 'andromeda'

galaxy_zoo_images_path = "../resources/galaxyzoo2/images_gz2/images/"

images_to_load = 1000
error_threshold = 0.0016
# error_threshold = 0.04
total_threads_number = 6

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    identified_objects = ic.start_comparison_process(images_parent_path + str(image_name + '.png'),
                                                     galaxy_zoo_images_path,
                                                     images_to_load=images_to_load,
                                                     zoom_from_center=25)

    if len(identified_objects) == 0:
        print("No similarities found on the catalog")
        exit()

    # Load the db files and search for a filename
    # db.load_dbs()
    # db.search_file(identified_objects)

    # plot.display_images(identified_objects, galaxy_zoo_images_path)

    ic.download_segmented_objects()
    # il.load_images(galaxy_zoo_images_path, 2000, 0)
    # il.load_image_prettified(images_parent_path + str(image_name + '.png'), download_segmented=False, display_images=True)

    # il.compare_filters("test")
