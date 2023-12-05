import keras
import tensorflow as tf
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, BatchNormalization, Activation, Dropout
from keras.models import Sequential

from keras.layers import Input


def custom_v3(number_of_pixels, classes):
    model = Sequential()
    model.add(Conv2D(64, (3, 3), padding='same', activation='relu', input_shape=(1, number_of_pixels, number_of_pixels)))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D())
    model.add(Dropout(0.25))

    model.add(Conv2D(128, (3, 3), padding='same', activation='relu'))
    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(MaxPooling2D())
    model.add(Dropout(0.25))

    model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D())

    model.add(Conv2D(32, (3, 3), padding='same', activation='relu'))
    model.add(Conv2D(32, (3, 3), activation='relu'))
    model.add(MaxPooling2D())

    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(classes, activation='softmax'))
    return model


def custom_v5(number_of_pixels, classes):
    model = Sequential()

    # First Convolutional Block
    model.add(Conv2D(64, (3, 3), input_shape=(1, number_of_pixels, number_of_pixels), padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Conv2D(64, (3, 3), padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    # Second Convolutional Block
    model.add(Conv2D(128, (3, 3), padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Conv2D(128, (3, 3), padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    # Third Convolutional Block
    model.add(Conv2D(256, (3, 3), padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Conv2D(256, (3, 3), padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    # Fourth Convolutional Block
    model.add(Conv2D(512, (3, 3), padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Conv2D(512, (3, 3), padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    # Flatten the feature maps
    model.add(Flatten())

    # First Fully Connected Layer
    model.add(Dense(512))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.2))

    # Second Fully Connected Layer
    model.add(Dense(1024))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.5))

    # Output Layer (Adjust the units based on the number of classes)
    model.add(Dense(classes, activation='softmax'))
    return model


def custom_v6(number_of_pixels, classes):
    model = Sequential()

    # First Convolutional Block
    model.add(Conv2D(512, (3, 3), input_shape=(1, number_of_pixels, number_of_pixels), padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Conv2D(512, (3, 3), padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    # Second Convolutional Block
    model.add(Conv2D(256, (3, 3), padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Conv2D(256, (3, 3), padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    # Third Convolutional Block
    model.add(Conv2D(128, (3, 3), padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Conv2D(128, (3, 3), padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    # Fourth Convolutional Block
    model.add(Conv2D(64, (3, 3), padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Conv2D(64, (3, 3), padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    # Flatten the feature maps
    model.add(Flatten())

    # First Fully Connected Layer
    model.add(Dense(512))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.2))

    # Second Fully Connected Layer
    model.add(Dense(1024))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.5))

    # Output Layer (Adjust the units based on the number of classes)
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