import numpy as np
from scipy.ndimage import gaussian_filter


astronomical_objects = []
objects_count = 0
# Used to filter out the images that are too small for usage
image_save_kernel_threshold = 7


def apply_filters(img, gaussian=False, grayscale=True):
    filtered_image = img

    # In case the values we have from the image are [0-255] we normalize them
    if filtered_image.max() > 1:
        filtered_image = filtered_image/255

    if grayscale:
        filtered_image = grayscale_filter(filtered_image)
    if gaussian:
        filtered_image = gaussian_filter(filtered_image, sigma=.1)

    return filtered_image


def grayscale_filter(img):
    return np.dot(img[...,:3], [0.2989, 0.5870, 0.1140])


def identify_and_outline_objects(img, plt=0, outline=True, save=True, zoom_from_center=15):
    global astronomical_objects, objects_count
    astronomical_objects = []
    objects_count = 0
    width, height = img.shape
    # Used to trigger the beginning of an astronomical body
    PIXEL_THRESHOLD = 0.7
    # Used to expand the kernel in search for an object boundaries
    KERNEL_THRESHOLD = 0.3
    visited = np.zeros(img.shape)
    for x in range(width-5):
        for y in range(height-5):
            # Checks if we identified a pixel from a celestial body or if it is part of another body already visited
            if img[x, y] > PIXEL_THRESHOLD and visited[x, y] == 0:
                # Get the center with the maximum local value
                center = get_center(img, x, y, kernel_size=5)
                kernel_size = 3
                # While the pixel is not out of bounds and its kernel value is in parameters we increase the kernel size
                # Increasing this size will let us expand the search and later expand the drawn circle
                while center[0]+kernel_size < width and center[1]+kernel_size < height and get_kernel_value(img, kernel_size, visited, center[0], center[1]) > KERNEL_THRESHOLD:
                    kernel_size += 2

                # Create the coords object to be drawn later from the maximum center found
                coords = [center[0], center[1]]
                snake = outline_object(coords, img, outline_size=kernel_size)

                # Will save the image for later use if the option is enabled or
                # if the kernel size is big enough
                # This is done so we won't take in consideration images that are too small
                # Hence resulting in one pixel segmented objects that will yield no results
                if save and kernel_size > image_save_kernel_threshold:
                    save_object_in_memory(img, coords, kernel_size, zoom_from_center)

                # Color the circle
                if outline and plt != 0:
                    circle_color = get_outline_color(img[x, y])
                    # Plotting the object boundary marker
                    # plt.plot(contour[:, 0], contour[:, 1], '-b', lw=1)
                    # Plotting the circle around face
                    plt.plot(snake[:, 0], snake[:, 1], circle_color, lw=1)


# Generates the coordinates of the circles that are going to be drawn
def outline_object(coords, img, outline_size=1):
    if outline_size < 1:
        outline_size = 1

    OUTLINE_NUMBER_OF_POINTS = 50

    # Generating a circle based on x1, x2
    # Localising the circle's center at 220, 110
    x1 = coords[1] + outline_size * np.cos(np.linspace(0, 2 * np.pi, OUTLINE_NUMBER_OF_POINTS))
    x2 = coords[0] + outline_size * np.sin(np.linspace(0, 2 * np.pi, OUTLINE_NUMBER_OF_POINTS))

    # Generating a circle based on x1, x2
    snake = np.array([x1, x2]).T
    # Computing the Active Contour for the given image
    # active_contour(img, snake)
    return snake


# Will return the color of the circle based on the intensity of the pixel
def get_outline_color(pixel_value):
    if pixel_value > .8:
        return '#660000'
    elif pixel_value > .7:
        return '#ff1919'
    elif pixel_value > .6:
        return '#ff6666'
    elif pixel_value > .5:
        return '#ff8000'
    else:
        return '#FFFF00'


# Gets the value of the whole kernel and sets the neighbours to visited
def get_kernel_value(img, kernel_size, visited, x, y):
    sum = 0
    step = int(kernel_size / 2)
    for i in range(-step, step+1):
        for j in range(-step, step+1):
            sum += img[x+i, y+j]
            visited[x+i, y+j] = 1

    return sum/(kernel_size * kernel_size)


# Gets the center based on a local maximum
# The local maximum is found in a matrix with the size of kernel_size
# The algorithm will continuously search for a maximum until one is found
def get_center(img, x, y, kernel_size=5):
    local_maximum = -1
    kernel_maximum = 0
    step = int(kernel_size/2)
    # Search the vicinity of the image while there is a better maximum
    while kernel_maximum > local_maximum:
        local_maximum = kernel_maximum
        kernel = img[x-step:x+step+1, y-step:y+step+1]
        for i in range(kernel_size):
            for j in range(kernel_size):
                if 0 < kernel.size < kernel_size and kernel[i, j] > kernel_maximum:
                    kernel_maximum = kernel[i, j]
                    x, y = x+i, y+j
    return x, y


# Save the objects identified on the local memory for easy later usage
# The object will be saved from the center with the kernel size identified previously
# distance_from_center parameter is used to save a kernel with this distance outside of the center of the object
# for eg. if the object has a size of 10x10 pixels, this will save that image along with another 15x15 pixels
# besides the object
def save_object_in_memory(img, center, kernel_size, zoom_from_center):
    global astronomical_objects, objects_count
    step = int(kernel_size/2)+zoom_from_center
    x, y = center[0], center[1]
    kernel = img[x - step:x + step + 1, y - step:y + step + 1]
    if kernel.size > 0:
        astronomical_objects.append(kernel)
        objects_count += 1
