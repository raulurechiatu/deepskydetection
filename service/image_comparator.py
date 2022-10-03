import threading
import time
from threading import Lock
import numba
# from numba import vectorize, cuda

from service import image_processor as ip
from service import image_loader as il
import numpy as np
from service import plot_builder as plot
from utils import progressbar

import cv2

mutex = Lock()


# A method used to start the comparison process and everything that is required to do that
# It will load the original image (the image we will test on), the catalog images and their names and will call the
# comparison algorithm
def start_comparison_process(original_image_path, catalog_images_path, total_threads_number, error_threshold,
                             images_to_load=-1, zoom_from_center=15):
    # Get the images from the catalog (locally for now)
    catalog_images, catalog_image_names = il.load_images(catalog_images_path, images_to_load, 0)

    # Load the original image the user wants to run the algorithm on
    original_image = il.load_image_matplot(original_image_path)
    original_image = ip.apply_filters(original_image, gaussian=False)
    ip.identify_and_outline_objects(original_image, outline=False, save=True, zoom_from_center=zoom_from_center)
    print("Segmentation process found ", ip.objects_count, " images.")

    return compare_images(ip.astronomical_objects, catalog_images, catalog_image_names, total_threads_number,
                          error_threshold)


# TODO: make both lists have the same size and start the comparison process
# This method will start the algorithm for comparison of two images
# It will return the name of the file for now
# It returns a map of the object identified in the database, the segment and the confidence level
def compare_images(segmented_images, catalog_images, catalog_image_names, total_threads_number, error_threshold):
    before = time.time()

    # valid_objects_files, mean_err, minimum_err = task_compare_images(segmented_images, catalog_images, catalog_image_names, error_threshold)
    valid_objects_files = []
    mean_err = 0
    minimum_err = 1
    threads = []
    results = []

    for thread in range(total_threads_number):
        threads.append(
            threading.Thread(target=task_compare_images, args=(segmented_images, catalog_images, catalog_image_names,
                                                               thread, results, total_threads_number, error_threshold)))

        # starting threads
        threads[thread].start()

    for thread in range(total_threads_number):
        # wait until all threads finish
        threads[thread].join()
        mean_err += results[thread][1]
        if len(results[thread][0]) > 0:
            valid_objects_files.extend(results[thread][0])
        if results[thread][2] < minimum_err:
            minimum_err = results[thread][2]

    print(valid_objects_files)
    after = time.time()
    mean_err = mean_err / (len(catalog_images) * len(segmented_images))
    print("Comparator service took", (after - before), "s and generated", len(valid_objects_files),
          "similarities with an error less than", error_threshold, ".\nClosest error to the threshold is:", minimum_err,
          ".\nThe mean value of the error along the dataset was ", mean_err)
    # valid_objects_files.append(catalog_image_names[catalog_image_result_id])
    return valid_objects_files


def task_compare_images(segmented_images, catalog_images, catalog_image_names, thread_number, results,
                        total_threads_number, error_threshold):
    valid_objects_files = []
    minimum_err = 1
    mean_err = 0
    catalog_images_len = len(catalog_images)
    chunck_size = catalog_images_len / total_threads_number
    start = int(chunck_size * thread_number)
    end = int(start + chunck_size)
    current_catalog_image = start
    current_catalog_image_counter = 0

    progressbar.printProgressBar(0, chunck_size, prefix='Thread ' + str(thread_number) + ':', suffix='Complete',
                                 length=50)

    for catalog_image in range(start, end):
        current_catalog_image += 1
        current_catalog_image_counter += 1
        progressbar.printProgressBar(current_catalog_image_counter, chunck_size,
                                     prefix='Thread ' + str(thread_number) + ':', suffix='Complete',
                                     length=50)

        # We pre process the catalog image too here in order to align with the one we segmented
        catalog_image_filtered = ip.apply_filters(catalog_images[catalog_image])
        current_segment = 0
        for segmented_image in segmented_images:
            current_segment += 1
            valid_objects_files, mean_err, minimum_err = compare(
                                                                 catalog_image_filtered, segmented_image,
                                                                 # ip.get_object_signature(catalog_image_filtered), ip.get_object_signature(segmented_image),
                                                                 catalog_image_names,
                                                                 current_catalog_image, mean_err, valid_objects_files,
                                                                 current_segment,
                                                                 minimum_err,
                                                                 error_threshold)  # logic for comparing catalogImages[i] to segmentedImages[i]

    mutex.acquire()
    results.append((valid_objects_files, mean_err, minimum_err))
    mutex.release()


def compare(catalog_image, segmented_image, catalog_image_names, current_catalog_image, mean_err,
            valid_objects_files, current_segment, minimum_err, error_threshold):
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

    return valid_objects_files, mean_err, minimum_err


@numba.jit
def mse(segmented_image, catalog_image):
    err = np.sum((segmented_image - catalog_image) ** 2)
    err /= float(segmented_image.shape[0] * catalog_image.shape[1])
    # return the MSE, the lower the error, the more "similar" the two images are
    return err


# @vectorize(['float32(float32, float32)',
#             'float64(float64, float64)'],
#            target='cuda')
# def mse2(segmented_image: np.ndarray, catalog_image: np.ndarray) -> float:
#     err = np.sum((segmented_image - catalog_image) ** 2)
#     err /= float(segmented_image.shape[0] * catalog_image.shape[1])
#     # return the MSE, the lower the error, the more "similar" the two images are
#     return err


# @vectorize(['float64(float64, float64)',
#             'float64(float64, float64)'],
#             target='cuda')
def test(image_a: np.ndarray, image_b: np.ndarray):
    return (image_a - image_b) ** 2


def mse2(image_a: np.ndarray, image_b: np.ndarray) -> float:
    err = np.sum(test(image_a, image_b))
    err /= float(image_a.shape[0] * image_b.shape[1])
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
    img = np.uint8(img * 255)
    kp = sft.detect(img, None)
    result = cv2.drawKeypoints(img, kp, img)
    plot.display_two_images(result, result)


def download_segmented_objects(image_name):
    il.download_segmented_objects(image_name)
