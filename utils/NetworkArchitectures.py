import keras
import tensorflow as tf
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from keras.models import Sequential

from keras.layers import Input


def simple_old(number_of_pixels):
    # model = tf.keras.Sequential([
    #     # tf.keras.layers.Rescaling(1. / 255, input_shape=(number_of_pixels, number_of_pixels)),
    #     tf.keras.layers.Flatten(input_shape=(number_of_pixels, number_of_pixels)),
    #     tf.keras.layers.Dense(128, activation='relu'),
    #     tf.keras.layers.Dense(3)
    # ])
    model = tf.keras.Sequential([
        # tf.keras.layers.Rescaling(1. / 255, input_shape=(number_of_pixels, number_of_pixels, 1)),
        tf.keras.layers.Conv2D(16, 3, padding='same', activation='relu', input_shape=(number_of_pixels, number_of_pixels, 1)),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Conv2D(32, 3, padding='same', activation='relu'),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Conv2D(64, 3, padding='same', activation='relu'),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(3)
    ])
    return model


def simple(number_of_pixels, classes):
    model = Sequential()
    model.add(Conv2D(32, (3, 3), padding='same', activation='relu', input_shape=(1, number_of_pixels, number_of_pixels)))
    model.add(Conv2D(32, (3, 3), activation='relu'))
    model.add(MaxPooling2D())
    model.add(Dropout(0.25))

    model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D())
    model.add(Dropout(0.25))

    # model.add(Conv2D(number_of_pixels*4, (3, 3), padding='same', activation='relu'))
    # model.add(Conv2D(number_of_pixels*4, (3, 3), activation='relu'))
    # model.add(MaxPooling2D())
    # model.add(Dropout(0.25))

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


def create_inception(number_of_pixels):
    inputs = Input(shape=(1, number_of_pixels, number_of_pixels))

    return tf.keras.applications.InceptionResNetV2(
        include_top=True,
        weights=None,
        input_tensor=inputs,
        input_shape=(1, number_of_pixels, number_of_pixels),
        pooling=None,
        classes=1,
        classifier_activation="sigmoid",
    )


# stripped down Vgg16 model functional implementation
def create_functional_vgg16(number_of_pixels):
    inputs = Input(shape=(1, number_of_pixels,  number_of_pixels))

    x = Conv2D(64, (3, 3), activation='relu', input_shape=(1, number_of_pixels,  number_of_pixels))(inputs)
    x = Conv2D(64, (3, 3), activation='relu')(x)
    x = MaxPooling2D(2, 2)(x)
    x = Conv2D(64, (3, 3), activation='relu')(x)
    x = Conv2D(64, (3, 3), activation='relu')(x)
    x = MaxPooling2D(2, 2)(x)
    x = Conv2D(64, (3, 3), activation='relu')(x)
    x = Conv2D(64, (3, 3), activation='relu')(x)
    x = Conv2D(64, (3, 3), activation='relu')(x)
    x = MaxPooling2D(2, 2)(x)
    #reshap = keras.layers.Reshape((28, 28))(hidden3)
    #concat_ = keras.layers.Concatenate()([inputs, reshap])
    #flatten2 = Flatten(input_shape=[28, 28])(concat_)
    x = Flatten()(x)
    x = Dense(100, activation='relu')(x)
    x = Dense(50, activation='relu')(x)
    output = Dense(1, activation='sigmoid')(x)
    model = keras.Model(inputs=[inputs], outputs=[output])

    # input shape for Conv2D is (batch_size{auto-fed when building() model/ calling model.fit()}, number of channels, img dimension x, img dimension y)
    # or (batch_size{auto-fed when building() model/ calling model.fit()}, img dimension x, img dimension y, number of channels) if used default configuration 'channels_last'

    # model.compile(loss=tf.losses.MeanSquaredError(),
    #               optimizer=tf.optimizers.Adagrad(),
    #               metrics=["accuracy"])

    return model
