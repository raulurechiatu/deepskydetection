import keras
import tensorflow as tf
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization, GlobalAveragePooling2D
from keras.models import Sequential
from keras.layers import add, ZeroPadding2D, concatenate, Conv2D, Attention
import keras.initializers.initializers_v1 as initializers

from keras.layers import Input


def custom_v3(number_of_pixels, classes):
    model = Sequential()
    model.add(Conv2D(64, (3, 3), padding='same', activation='relu', input_shape=(1, number_of_pixels, number_of_pixels)))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D())
    model.add(Dropout(0.1))

    model.add(Conv2D(128, (3, 3), padding='same', activation='relu'))
    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(MaxPooling2D())
    model.add(Dropout(0.2))

    model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D())

    model.add(Conv2D(32, (3, 3), padding='same', activation='relu'))
    model.add(Conv2D(32, (3, 3), activation='relu'))
    model.add(MaxPooling2D())

    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.25))
    model.add(Dense(classes, activation='softmax'))
    return model


def custom_v2(number_of_pixels, classes):
    # Define the custom CNN architecture for grayscale images
    model = Sequential()

    # Convolutional Layer 1 with Batch Normalization and Max-Pooling
    model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer=initializers.HeNormal(),
                            input_shape=(1, number_of_pixels, number_of_pixels)))
    model.add(BatchNormalization())
    model.add(MaxPooling2D((2, 2)))

    # Convolutional Layer 2 with Batch Normalization and Max-Pooling
    model.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer=initializers.HeNormal()))
    model.add(BatchNormalization())
    model.add(MaxPooling2D((2, 2)))

    # Convolutional Layer 3 with Batch Normalization and Global Average Pooling
    model.add(Conv2D(128, (3, 3), activation='relu', kernel_initializer=initializers.HeNormal()))
    model.add(BatchNormalization())
    model.add(GlobalAveragePooling2D())

    # Fully Connected Layers with Dropout
    model.add(Dense(256, activation='relu', kernel_initializer=initializers.HeNormal()))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))
    model.add(Dense(128, activation='relu', kernel_initializer=initializers.HeNormal()))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))

    model.add(Dense(classes, activation='softmax'))

    return model


def custom_v4(number_of_pixels, classes):
    # Define the custom CNN architecture for grayscale images
    model = Sequential()

    # Convolutional Layer 1 with Max-Pooling
    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(1, number_of_pixels, number_of_pixels)))
    model.add(MaxPooling2D((2, 2)))

    # Convolutional Layer 2 with Max-Pooling
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2)))

    # Convolutional Layer 3 with Global Average Pooling
    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(GlobalAveragePooling2D())

    # Fully Connected Layers
    model.add(Dense(64, activation='relu'))

    model.add(Dense(classes, activation='softmax'))

    return model


def custom_v6(number_of_pixels, classes):
    # Input layer (assuming grayscale images)
    input_shape = (1, number_of_pixels, number_of_pixels)

    # Define the improved CNN architecture
    model = Sequential()

    # Convolutional Layer 1
    model.add(Conv2D(64, (5, 5), activation='relu', input_shape=input_shape))
    model.add(MaxPooling2D((2, 2)))

    # Convolutional Layer 2
    model.add(Conv2D(128, (5, 5), activation='relu'))
    model.add(MaxPooling2D((2, 2)))

    # Convolutional Layer 3
    model.add(Conv2D(256, (5, 5), activation='relu'))
    model.add(GlobalAveragePooling2D())

    # Fully Connected Layers
    model.add(Dense(128, activation='relu'))
    model.add(Dense(64, activation='relu'))

    # Output Layer
    model.add(Dense(classes, activation='softmax'))

    return model


def custom_v5(number_of_pixels, classes):
    # Input layer (assuming grayscale images)
    input_shape = (1, number_of_pixels, number_of_pixels)

    # Define the improved CNN architecture
    model = Sequential()

    # model.add(Conv2D(32, (5, 5), activation='relu', input_shape=input_shape))
    # model.add(BatchNormalization())
    # model.add(MaxPooling2D((2, 2)))

    # Convolutional Layer 1
    # model.add(Conv2D(32, (5, 5), activation='relu', input_shape=input_shape))

    # Convolutional Layer 1
    model.add(Conv2D(64, (5, 5), activation='relu', input_shape=input_shape))
    model.add(BatchNormalization())
    model.add(MaxPooling2D((2, 2)))
    model.add(Dropout(0.3))

    # Convolutional Layer 2
    model.add(Conv2D(128, (5, 5), activation='relu'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Dropout(0.2))

    # Convolutional Layer 3
    model.add(Conv2D(256, (5, 5), activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D((2, 2)))
    model.add(Dropout(0.1))

    # Convolutional Layer 3
    # model.add(Conv2D(512, (5, 5), activation='relu'))
    # model.add(MaxPooling2D((2, 2)))

    # Convolutional Layer 3
    # model.add(Conv2D(256, (5, 5), activation='relu'))
    # model.add(BatchNormalization())

    model.add(GlobalAveragePooling2D())

    # Convolutional Layer 4
    # model.add(Conv2D(512, (5, 5), activation='relu'))
    # model.add(BatchNormalization())
    # model.add(GlobalAveragePooling2D())

    # Fully Connected Layers
    model.add(Dense(256, activation='relu'))
    model.add(Dense(128, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.1))

    # Output Layer
    model.add(Dense(classes, activation='softmax'))

    return model


def custom_v1(number_of_pixels, classes):
    model = Sequential()
    model.add(Conv2D(32, (3, 3), padding='same', activation='relu', input_shape=(1, number_of_pixels, number_of_pixels)))
    model.add(Conv2D(32, (3, 3), activation='relu'))
    model.add(MaxPooling2D())
    model.add(Dropout(0.25))

    model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D())
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(classes, activation='softmax'))
    return model


# Keras ResNet50V2 model
def create_ResNet50V2(number_of_pixels, classes=3):
    inputs = Input(shape=(1, number_of_pixels, number_of_pixels))

    return tf.keras.applications.ResNet50V2(
        include_top=True,
        weights=None,
        input_tensor=inputs,
        input_shape=(1, number_of_pixels, number_of_pixels),
        pooling=None,
        classes=classes,
        classifier_activation="softmax",
    )


def create_inception(number_of_pixels, classes):
    inputs = Input(shape=(1, number_of_pixels, number_of_pixels))

    return tf.keras.applications.InceptionResNetV2(
        include_top=True,
        weights=None,
        input_tensor=inputs,
        input_shape=(1, number_of_pixels, number_of_pixels),
        pooling=None,
        classes=classes,
        classifier_activation="softmax",
    )


def create_vgg16(number_of_pixels, classes):
    inputs = Input(shape=(1, number_of_pixels, number_of_pixels))

    return tf.keras.applications.VGG16(
        include_top=True,
        weights=None,
        input_tensor=inputs,
        input_shape=(1, number_of_pixels, number_of_pixels),
        pooling=None,
        classes=classes,
        classifier_activation="softmax",
    )


def create_mobile(number_of_pixels, classes):
    inputs = Input(shape=(1, number_of_pixels, number_of_pixels))

    return tf.keras.applications.MobileNet(
        include_top=True,
        weights=None,
        input_tensor=inputs,
        input_shape=(1, number_of_pixels, number_of_pixels),
        pooling=None,
        classes=classes,
        classifier_activation="softmax",
    )


def create_nasnet(number_of_pixels, classes):
    inputs = Input(shape=(1, number_of_pixels, number_of_pixels))

    return tf.keras.applications.NASNetMobile(
        include_top=True,
        weights=None,
        input_tensor=inputs,
        input_shape=(1, number_of_pixels, number_of_pixels),
        pooling=None,
        classes=classes,
        classifier_activation="softmax",
    )


def create_densenet(number_of_pixels, classes):
    inputs = Input(shape=(1, number_of_pixels, number_of_pixels))

    return tf.keras.applications.DenseNet121(
        include_top=True,
        weights=None,
        input_tensor=inputs,
        input_shape=(1, number_of_pixels, number_of_pixels),
        pooling=None,
        classes=classes,
        classifier_activation="softmax",
    )


def create_efficient(number_of_pixels, classes):
    inputs = Input(shape=(1, number_of_pixels, number_of_pixels))

    return tf.keras.applications.EfficientNetB5(
        include_top=True,
        weights=None,
        input_tensor=inputs,
        input_shape=(1, number_of_pixels, number_of_pixels),
        pooling=None,
        classes=classes,
        classifier_activation="softmax",
    )