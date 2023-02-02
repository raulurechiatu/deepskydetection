import gc
import os

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd
from keras.utils import to_categorical
from sklearn import metrics
from sklearn.model_selection import train_test_split
from tensorflow import keras
from tensorflow.keras.optimizers import SGD
from utils import NetworkArchitectures


number_of_pixels = 64


def train_model(data_train, data_test, labels_train, labels_test, data_validate, labels_validate):
    # configuring keras backend format for channel position
    keras.backend.set_image_data_format('channels_first')
    print(keras.backend.image_data_format())
    batch_size = 16

    print(data_train.shape)
    # print(data_train[0])
    # print(labels_train.shape)
    print(len(labels_train))
    print(data_train.shape)

    # model = arch.get(item[0])()  # don't forget '()' to call the function
    # model = NetworkArchitectures.simple(number_of_pixels)
    model = NetworkArchitectures.create_ResNet50V2(number_of_pixels)

    # model.summary()

    opt = SGD(learning_rate=0.05)
    # if RELOAD_WEIGHTS:
    #     model.load_weights("models/" + WEIGHTS_FILE_NAME)

    model.compile(
        optimizer=tf.optimizers.Adam(),
        # optimizer=opt,
        # loss=tf.losses.BinaryCrossentropy(),
        loss=tf.losses.CategoricalCrossentropy(),
        # loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        # loss=tf.losses.MeanSquaredError(),
        metrics=[keras.metrics.Accuracy()]
    )

    result = model.fit(data_train,
                       labels_train,
                       epochs=5,
                       validation_data=(data_test, labels_test),
                       batch_size=batch_size
                       # callbacks=[es]
                       # callbacks=[metrics]
                       # shuffle = True # optional parameter for composites only
                       )
    # save model
    # model.save('models/model_' + str(item[0]) + '_' + str(item[1]) + '_' + str(item[2]) + '_' + str(item[3]) + '.h5')
    # model.save('models/' + WEIGHTS_FILE_NAME)

    # original precision eval implementation
    test_loss, test_acc = model.evaluate(data_test, labels_test, verbose=1)

    print('\nTest loss:', test_loss)
    print('\nTest accuracy:', test_acc)
    print('\nMetric names:', model.metrics_names)

    # for binary classification
    # test_prediction = (model.predict(data_test) > 0.5).astype("int32")

    # for multi-class classification
    test_prediction = np.argmax(model.predict(data_test), axis=-1)

    print("data_test.shape: ", data_test.shape)
    print("labels_test.shape: ", labels_test.shape)
    print("test_prediction.shape: ", test_prediction.shape)
    print("test_prediction: ", test_prediction)
    actual_vals = []
    correct_predictions = []
    for label_id in range(len(labels_test)):
        actual_vals.append(np.where(labels_test[label_id] > 0.5)[0][0])
        correct_predictions.append(actual_vals[label_id] == test_prediction[label_id])
    print("test_actual: ", actual_vals)
    print("correct prediction: ", correct_predictions)
    print("computed accuracy: ", sum(bool(x) for x in correct_predictions) / len(correct_predictions))

    # print("recall w param funciton micro: ",
    #       metrics.recall_score(labels_test, test_prediction, average='micro', zero_division=1))
    # print("recall w param funciton macro: ",
    #       metrics.recall_score(labels_test, test_prediction, average='macro', zero_division=1))
    # print("recall w param funciton weighted: ",
    #       metrics.recall_score(labels_test, test_prediction, average='weighted', zero_division=1))
    # print("recall w param funciton binary: ",
    #       metrics.recall_score(labels_test, test_prediction, average=None, zero_division=1))

    # df = pd.DataFrame(result.history)
    # df.plot(figsize=(8, 5))
    # plt.grid(True)
    # plt.gca().set_ylim(0, 1)
    # plt.show()


    # save model
    # model.save('models/model_' + str(item[0]) + '_' + str(item[1]) + '_' + str(item[2]) + '_' + str(item[3]) + '.h5')
    # model.save('models/' + WEIGHTS_FILE_NAME)

    # reset model
    keras.backend.clear_session()


def configure_gpu():
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            # tf.config.set_visible_devices(gpus[0], 'GPU')
            # logical_gpus = tf.config.list_logical_devices('GPU')
            # print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPU")
            tf.config.experimental.set_memory_growth(gpus[0], True)
            tf.config.set_logical_device_configuration(
                gpus[0],
                [tf.config.LogicalDeviceConfiguration(memory_limit=6000)]
            )
        except RuntimeError as e:
            # Memory growth must be set before GPUs have been initialized
            print("Error: ".e)


def train(galaxy_images, nebulae_images, star_images):
    # (60000, 28, 28)
    # (60000,)

    labels = [0] * len(galaxy_images) + [1] * len(nebulae_images) + [2] * len(star_images)
    # labels = np.array(labels)
    # labels = labels.reshape(-1, 1)
    labels = to_categorical(labels, 3)

    images = np.append(galaxy_images, nebulae_images, axis=0)
    images = np.append(images, star_images, axis=0) / 255.0
    print(labels.shape)
    print(images.shape)

    data_train, data_test, labels_train, labels_test = train_test_split(images,
                                                                        labels,
                                                                        test_size=0.1,
                                                                        shuffle=True,
                                                                        random_state=1)
    del images, labels
    validation_set_size = 50
    data_validate = data_train[-validation_set_size:]
    labels_validate = labels_train[-validation_set_size:]
    data_train = data_train[:-validation_set_size]
    labels_train = labels_train[:-validation_set_size]

    # reshape data for model compatibility
    data_train = data_train.reshape(-1, 1, number_of_pixels, number_of_pixels)
    data_test = data_test.reshape(-1, 1, number_of_pixels, number_of_pixels)
    data_validate = data_validate.reshape(-1, 1, number_of_pixels, number_of_pixels)

    # data_train = tf.expand_dims(data_train, axis=-1)
    # data_test = tf.expand_dims(data_test, axis=-1)
    # data_validate = tf.expand_dims(data_validate, axis=-1)

    train_model(data_train, data_test, labels_train, labels_test, data_validate, labels_validate)


