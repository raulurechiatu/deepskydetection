import matplotlib.pyplot as plt
from pathlib import Path
import matplotlib.image as mpimg
from segmentation import custom_processor as cp
from utils import image_processor as ip
import numpy as np
import cv2 as cv
from statistics import mean

def display_images(identified_objects, catalog_images_path):
    for identified_object in identified_objects:
        segment = identified_object['segment']
        path = Path(__file__).parent / catalog_images_path / identified_object['filename']
        catalog_image = mpimg.imread(path)

        f, axarr = plt.subplots(1, 2)
        plt.suptitle('Segment ' + str(identified_object['segment']) + '-' + identified_object['filename'] +
                     ' (Error value ' + str(round(identified_object['err'], 5)) + ')')
        axarr[0].imshow(cp.astronomical_objects[segment-1], cmap='gray')

        catalog_image = ip.apply_filters(catalog_image)
        axarr[1].imshow(catalog_image, cmap='gray')

        plt.show()


def display_images_one_window(identified_objects, catalog_images_path):
    row_index = 0
    col_index = 0
    split_point = int(np.floor(np.sqrt(len(identified_objects))))
    row_reset_count = 0
    # Display the comparison between images
    f, axarr = plt.subplots(split_point+1, split_point*2)
    plt.tight_layout()
    for identified_object in identified_objects:
        segment = identified_object['segment']
        path = Path(__file__).parent / catalog_images_path / identified_object['filename']
        catalog_image = mpimg.imread(path)

        # f, axarr = plt.subplots(1, 2)
        plt.suptitle('Segment ' + str(identified_object['segment']) + '-' + identified_object['filename'] +
                     ' (Error value ' + str(identified_object['err']) + ')')
        axarr[row_index, col_index].imshow(ip.astronomical_objects[segment], cmap='gray')

        catalog_image = ip.apply_filters(catalog_image)
        axarr[row_index, col_index+1].imshow(catalog_image, cmap='gray')

        row_reset_count += 1

        col_index += 2
        if row_reset_count >= split_point:
            row_index += 1
            col_index = 0
            row_reset_count = 0

    plt.show()


def display_two_images(imageA, imageB, title=''):
    f, axarr = plt.subplots(1, 2)
    plt.suptitle(title)
    axarr[0].imshow(imageA, cmap='gray')
    axarr[1].imshow(imageB, cmap='gray')
    plt.show()


def display_rotations(images, image_index, title=''):
    f, axarr = plt.subplots(3, 4)
    plt.suptitle(title)
    axarr[0][0].imshow(images[image_index + 0], cmap='gray')
    axarr[0][1].imshow(images[image_index + 1], cmap='gray')
    axarr[0][2].imshow(images[image_index + 2], cmap='gray')
    axarr[0][3].imshow(images[image_index + 3], cmap='gray')
    axarr[1][0].imshow(images[image_index + 4], cmap='gray')
    axarr[1][1].imshow(images[image_index + 5], cmap='gray')
    axarr[1][2].imshow(images[image_index + 6], cmap='gray')
    axarr[1][3].imshow(images[image_index + 7], cmap='gray')
    axarr[2][0].imshow(images[image_index + 8], cmap='gray')
    axarr[2][1].imshow(images[image_index + 9], cmap='gray')
    axarr[2][2].imshow(images[image_index + 10], cmap='gray')
    axarr[2][3].imshow(images[image_index + 11], cmap='gray')
    plt.show()


def display_image(image, title=''):
    # get_circles(image)
    f, axarr = plt.subplots(1)
    plt.suptitle(title)
    axarr.imshow(image, cmap='gray')
    plt.show()


def display_image_prediction(image, title_1='', title_2=''):
    # get_circles(image)
    f, axarr = plt.subplots(1)
    plt.suptitle(title_1)
    axarr.imshow(image, cmap='gray')
    plt.show()


def get_circles(img):
    hh, ww = img.shape[:2]
    min_dist = int(ww/10)
    circles = cv.HoughCircles(img, cv.HOUGH_GRADIENT, 1, minDist=min_dist, param1=50, param2=10, minRadius=0,
                               maxRadius=0)
    # draw circles
    result = img.copy()
    for circle in circles[0]:
        # draw the circle in the output image
        (x, y, r) = circle
        x = int(x)
        y = int(y)
        r = int(r/4)
        cv.circle(result, (x, y), r, (255, 255, 255), 1)
    return result


def plot_roc_curve(fpr, tpr, number_of_classes):
    # plotting
    fpr_mean = mean(fpr)
    tpr_mean = mean(tpr)
    colors = get_random_colors()

    for i in range(number_of_classes):
        label = 'Class ' + str(i)
        plt.plot(fpr[i], tpr[i], linestyle='solid', color=colors[i], label=label)

    # plt.plot(fpr[0], tpr[0], linestyle='solid', color='orange', label='Class 0 vs Rest')
    # plt.plot(fpr[1], tpr[1], linestyle='solid', color='green', label='Class 1 vs Rest')
    # plt.plot(fpr[2], tpr[2], linestyle='solid', color='blue', label='Class 2 vs Rest')
    # plt.plot(fpr[3], tpr[3], linestyle='solid', color='red', label='Class 3 vs Rest')
    # plt.plot(fpr[4], tpr[4], linestyle='solid', color='yellow', label='Class 4 vs Rest')
    plt.plot(fpr_mean, tpr_mean, linestyle='--', color='darkblue', label='Average ROC Curve')
    plt.axline((0, 0), (1, 1), linestyle='--', color='gray')
    plt.xlim(right=1, left=-0.02)
    plt.ylim(top=1.02, bottom=-0.02)
    plt.title('Multiclass ROC curve')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive rate')
    plt.legend(loc='best')
    plt.savefig('Multiclass ROC', dpi=300)
    plt.show()


def get_random_colors():
    clrs = np.linspace(0, 1, 18)
    np.random.shuffle(clrs)
    colors = []
    for i in range(0, 72, 4):
        idx = np.arange( 0, 18, 1 )
        np.random.shuffle(idx)
        r = clrs[idx[0]]
        g = clrs[idx[1]]
        b = clrs[idx[2]]
        a = clrs[idx[3]]
        colors.append([r, g, b, a])
    return colors
