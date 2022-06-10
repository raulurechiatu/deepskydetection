import time
from service import image_processor as ip
from service import image_loader as il
import numpy as np


# A method used to start the comparison process and everything that is required to do that
# It will load the original image (the image we will test on), the catalog images and their names and will call the
# comparison algorithm
def start_comparison_process(original_image_path, catalog_images_path, images_to_load=-1, error_threshold=0.05, distance_from_center=15):
    # Get the images from the catalog (locally for now)
    catalog_images, catalog_image_names = il.load_images(catalog_images_path, images_to_load, 0)

    # Load the original image the user wants to run the algorithm on
    original_image = il.load_image_matplot(original_image_path)
    original_image = ip.apply_filters(original_image, gaussian=False)
    ip.identify_and_outline_objects(original_image, outline=False, save=True, distance_from_center=distance_from_center)

    return compare_images(ip.astronomical_objects, catalog_images, catalog_image_names, error_threshold)


# TODO: make both lists have the same size and start the comparison process
# This method will start the algorithm for comparison of two images
# It will return the name of the file for now
# Later we'll have it TODO: return a map of images names with a confidence level above of a certain threshold
def compare_images(segmented_images, catalog_images, catalog_image_names, error_threshold):

    before = time.time()

    valid_objects_files = []
    catalog_image_result_id = 0
    current_catalog_image = 0
    minimum_err = 1

    # Iterate through alll the images available from the catalog and the segmented images we got from the original image
    for catalog_image in catalog_images:
        current_catalog_image += 1
        # We pre process the catalog image too here in order to align with the one we segmented
        catalog_image = ip.apply_filters(catalog_image)
        current_segment = 0

        for segmented_image in segmented_images:
            # Resize the images from the catalog to be the same size as the segmented ones
            catalog_image_new = np.resize(catalog_image, segmented_image.shape)
            # For now we use mse as this is a simple enough algorithm for a first version of the application
            err = mse(segmented_image, catalog_image_new)
            # If the error is lesser than the threshold we set up, save the image name
            if err < error_threshold:
                valid_object = {
                    "segment": current_segment,
                    "filename": catalog_image_names[current_catalog_image],
                    "err": err
                }
                valid_objects_files.append(valid_object)

            # Save the best minimum in order to better debug and understand where we are situated
            if err < minimum_err:
                minimum_err = err

            current_segment += 1

    print(valid_objects_files)
    after = time.time()
    print("Comparator service took", (after-before), "s and generated", len(valid_objects_files),
          "similarities with an error less than", error_threshold, ". Closest error to the threshold is:", minimum_err)
    # valid_objects_files.append(catalog_image_names[catalog_image_result_id])
    return valid_objects_files


def mse(segmented_image, catalog_image):
    err = np.sum((segmented_image.astype("float") - catalog_image.astype("float")) ** 2)
    err /= float(segmented_image.shape[0] * catalog_image.shape[1])
    # return the MSE, the lower the error, the more "similar" the two images are
    return err


