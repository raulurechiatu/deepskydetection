import service.image_comparator as ic
import service.db_manager as db

# Path for the initial image to start the algorithm on
# This image is for example a picture took with a personal telescope or obtain via the internet to be analyzed
images_parent_path = "../images/deepsky/"
image_name = 'andromeda'

galaxy_zoo_images_path = "../resources/galaxyzoo2/images_gz2/images/"

images_to_load=1000


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    image_name = ic.start_comparison_process(images_parent_path + str(image_name + '.png'), galaxy_zoo_images_path,
                                             images_to_load=images_to_load)
    db.load_dbs()
    db.search_file(image_name)
    # il.load_images(galaxy_zoo_images_path, 2000, 0)
    # il.load_image_prettified(images_parent_path + str(image_name + '.png'), download_segmented=False, display_images=True)

    # il.compare_filters("test")
