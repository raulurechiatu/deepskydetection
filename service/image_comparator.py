from service import image_processor as ip
from service import image_loader as il


# A method used to start the comparison process and everything that is required to do that
# It will load the original image (the image we will test on), the catalog images and their names and will call the
# comparison algorithm
def start_comparison_process(original_image_path, catalog_images_path, images_to_load=-1):
    # Get the images from the catalog (locally for now)
    catalog_images, catalog_image_names = il.load_images(catalog_images_path, images_to_load, 0)

    # Load the original image the user wants to run the algorithm on
    original_image = il.load_image_matplot(original_image_path)
    original_image = ip.apply_filters(original_image, gaussian=False)
    ip.identify_and_outline_objects(original_image, outline=False, save=True)

    return compare_images(ip.astronomical_objects, catalog_images, catalog_image_names)


# TODO: make both lists have the same size and start the comparison process
# This method will start the algorithm for comparison of two images
# It will return the name of the file for now
# Later we'll have it TODO: return a map of images names with a confidence level above of a certain threshold
def compare_images(original_images, catalog_images, catalog_image_names):
    # print(original_images[0], max(catalog_images[0]))
    catalog_image_result_id = 0
    return catalog_image_names[catalog_image_result_id]