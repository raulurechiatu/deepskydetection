import time
from service import image_processor as ip
from service import image_loader as il
import numpy as np
from service import plot_builder as plot

import cv2


# A method used to start the comparison process and everything that is required to do that
# It will load the original image (the image we will test on), the catalog images and their names and will call the
# comparison algorithm
def start_comparison_process(original_image_path, catalog_images_path, images_to_load=-1, error_threshold=0.05, zoom_from_center=15):
    # Get the images from the catalog (locally for now)
    catalog_images, catalog_image_names = il.load_images(catalog_images_path, images_to_load, 0)

    # Load the original image the user wants to run the algorithm on
    original_image = il.load_image_matplot(original_image_path)
    original_image = ip.apply_filters(original_image, gaussian=False)
    ip.identify_and_outline_objects(original_image, outline=False, save=True, zoom_from_center=zoom_from_center)

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
    mean_err = 0

    # Iterate through all the images available from the catalog and the segmented images we got from the original image
    for catalog_image in catalog_images:
        current_catalog_image += 1
        # We pre process the catalog image too here in order to align with the one we segmented
        catalog_image = ip.apply_filters(catalog_image)
        current_segment = 0

        for segmented_image in segmented_images:
            # Resize the images from the catalog to be the same size as the segmented ones
            # catalog_image_new = cv2.resize(catalog_image, segmented_image.shape, interpolation=cv2.INTER_AREA)
            segmented_image_new = cv2.resize(segmented_image, catalog_image.shape, interpolation=cv2.INTER_AREA)

            # For now we use mse as this is a simple enough algorithm for a first version of the application
            err = mse(segmented_image_new, catalog_image)
            mean_err += err
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
    mean_err = mean_err / len(catalog_images) * len(segmented_images)
    print("Comparator service took", (after-before), "s and generated", len(valid_objects_files),
          "similarities with an error less than", error_threshold, ".\nClosest error to the threshold is:", minimum_err,
          ".\nThe mean value of the error along the dataset was ", mean_err)
    # valid_objects_files.append(catalog_image_names[catalog_image_result_id])
    return valid_objects_files


def mse(segmented_image, catalog_image):
    err = np.sum((segmented_image.astype("float") - catalog_image.astype("float")) ** 2)
    err /= float(segmented_image.shape[0] * catalog_image.shape[1])
    # return the MSE, the lower the error, the more "similar" the two images are
    return err


# Next 2 algorithms are more advanced image comparison techniques as they use key point detecion
# Harris corner detector
def hcd(img):
    dst = cv2.cornerHarris(np.float32(img), 2, 3, 0.04)
    img[dst > 0.01 * dst.max()] = 1
    plot.display_two_images(img, img)


# Scale-invariant feature transform
def sift(img):
    sft = cv2.SIFT_create()
    img = np.uint8(img*255)
    kp = sft.detect(img, None)
    result = cv2.drawKeypoints(img, kp, img)
    plot.display_two_images(result, result)


def download_segmented_objects():
    il.download_segmented_objects()
