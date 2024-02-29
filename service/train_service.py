import gc
import os

import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report
import pandas as pd
from keras.utils import to_categorical
from sklearn import metrics
from sklearn.model_selection import train_test_split
from tensorflow import keras
from keras.optimizers import SGD
from collections import Counter
from service import plot_builder
from service.data_service import f1_m
from keras.utils import plot_model

from utils import NetworkArchitectures

# The image is firstly cropped to this size and after this is resized to the next value
crop_size = 180
# Original size 414
number_of_pixels = 64

# MODEL_SAVE_NAME = "L_CUSTOM_TEST_2_" + str(number_of_pixels) + "_"
# MODEL_SAVE_NAME = "L_RESNET_1_" + str(number_of_pixels) + "_"
MODEL_SAVE_NAME = "L_CUSTOM_1_" + str(number_of_pixels) + "_"
# MODEL_SAVE_NAME = "L_CUSTOM_2_2_64_60000_10ep"
model = None


def train_model(data_train, data_test, labels_train, labels_test, data_validate, labels_validate, number_of_classes):
    # configuring keras backend format for channel position
    global MODEL_SAVE_NAME, model
    keras.backend.set_image_data_format('channels_first')
    # model = NetworkArchitectures.create_ResNet50V2(number_of_pixels, number_of_classes)
    model = NetworkArchitectures.custom_v1(number_of_pixels, number_of_classes)
    # model = NetworkArchitectures.custom_v3(number_of_pixels, number_of_classes)
    # model = NetworkArchitectures.custom_v5(number_of_pixels, number_of_classes)
    # model = NetworkArchitectures.custom_v6(number_of_pixels, number_of_classes)

    model.compile(
        optimizer=tf.optimizers.Adam(),
        # optimizer=SGD(learning_rate=0.0001),
        # loss=tf.losses.BinaryCrossentropy(),
        loss=tf.losses.CategoricalCrossentropy(),
        # loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        # loss=tf.losses.MeanSquaredError(),
        metrics=[keras.metrics.CategoricalAccuracy(), keras.metrics.Precision(), keras.metrics.Recall(), f1_m,
                 keras.metrics.AUC()]
    )

    epochs = 15
    result = model.fit(data_train,
                       labels_train,
                       epochs=epochs,
                       validation_data=(data_test, labels_test),
                       batch_size=16
                       # callbacks=[es]
                       # callbacks=[metrics]
                       # shuffle = True # optional parameter for composites only
                       )

    # save model
    MODEL_SAVE_NAME += "_" + str(epochs) + "ep"
    model.save('models/10class/' + MODEL_SAVE_NAME + ".h5")

    # original precision eval implementation
    loss, acc, prec, rec, f1, auc = model.evaluate(data_test, labels_test, verbose=1)

    print('\nTest loss:', loss)
    print('\nTest accuracy:', acc)
    print('\nMetric names:', model.metrics_names)

    predictions = model.predict(data_test)
    # print(predictions)
    test_prediction = np.argmax(predictions, axis=-1)

    print("data_test.shape: ", data_test.shape)
    # print("test_prediction.shape: ", test_prediction.shape)
    print("test_prediction: ", test_prediction)
    actual_vals = []
    correct_predictions = []
    for label_id in range(len(labels_test)):
        actual_val = np.where(labels_test[label_id] > 0.5)[0][0]
        actual_vals.append({test_prediction[label_id], actual_val})
        correct_predictions.append(actual_val == test_prediction[label_id])
    print("test_actual: ", actual_vals)
    print("correct prediction: ", correct_predictions)
    print("computed accuracy: ", sum(bool(x) for x in correct_predictions) / len(correct_predictions))

    # reset model
    keras.backend.clear_session()


def configure_gpu():
    gpus = tf.config.list_physical_devices('GPU')
    print(gpus)
    print(len(gpus))

    if gpus:
        try:
            # tf.config.set_visible_devices(gpus[0], 'GPU')
            # logical_gpus = tf.config.list_logical_devices('GPU')
            # print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPU")
            tf.config.experimental.set_memory_growth(gpus[0], True)
            # tf.config.set_logical_device_configuration(
            #     gpus[0],
            #     [tf.config.LogicalDeviceConfiguration(memory_limit=6000)]
            # )
        except RuntimeError as e:
            # Memory growth must be set before GPUs have been initialized
            print("Error: ".e)


def get_model(model_name=None, custom_metrics=False):
    if model_name is None:
        model_name = MODEL_SAVE_NAME + ".h5"
    global model
    if model is None:
        model = keras.models.load_model("models/" + model_name, custom_objects={"f1_m": f1_m})
    if custom_metrics:
        model.compile(
            optimizer=tf.optimizers.Adam(),
            # optimizer=SGD(learning_rate=0.0001),
            # loss=tf.losses.BinaryCrossentropy(),
            loss=tf.losses.CategoricalCrossentropy(),
            # loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            # loss=tf.losses.MeanSquaredError(),
            metrics=[keras.metrics.CategoricalAccuracy(), keras.metrics.Precision(), keras.metrics.Recall(), f1_m,
                     keras.metrics.AUC()]
        )
    plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)
    return model


def evaluate(images, labels, model_name=None, manual=False):
    global model
    model = get_model(model_name, True)

    images = images.reshape(-1, 1, number_of_pixels, number_of_pixels)

    number_of_classes = len(set(labels))
    if not manual and labels is not None:
        categorical_labels = to_categorical(labels, number_of_classes)
        loss, acc, prec, rec, f1, auc = model.evaluate(images, categorical_labels, verbose=1)
        print(f"Loss: {loss}, acc: {acc}, precision: {prec}, recall: {rec}, f1: {f1}, auc: {auc}")

    # for multi-class classification
    scores = model.predict(images)
    test_prediction = np.argmax(scores, axis=-1)

    if labels is not None:
        print(classification_report(labels, test_prediction))
        correct_predictions = []
        prediction_mistakes = []
        results = []
        for label_id in range(len(labels)):
            correct_predictions.append(labels[label_id] == test_prediction[label_id])
            if labels[label_id] != test_prediction[label_id]:
                prediction_mistakes.append((labels[label_id], test_prediction[label_id]))
            results.append((labels[label_id], test_prediction[label_id]))
        print("correct prediction: ", correct_predictions)
        print("prediction mistakes ", str(len(prediction_mistakes)), " (expected, actual): ", prediction_mistakes)
        print("computed accuracy: ", sum(bool(x) for x in correct_predictions) / len(correct_predictions))

    print("data_test.shape: ", images.shape)
    print("test_prediction.shape: ", test_prediction.shape)
    # print("test_prediction: ", test_prediction)
    # roc curve for classes
    fpr = {}
    tpr = {}
    thresh = {}

    for i in range(number_of_classes):
        fpr[i], tpr[i], thresh[i] = metrics.roc_curve(labels, scores[:,i], pos_label=i)

    plot_builder.plot_roc_curve(fpr, tpr, number_of_classes)

    return results
    # df = pd.DataFrame(result.history)
    # df.plot(figsize=(8, 5))
    # plt.grid(True)
    # plt.gca().set_ylim(0, 1)
    # plt.show()


def evaluate_image(img):
    final_images = img.reshape(-1, 1, number_of_pixels, number_of_pixels)
    return np.argmax(get_model().predict(final_images, verbose=0), axis=-1)


def train(images, labels, image_names):
    global MODEL_SAVE_NAME
    # labels = [0] * len(galaxy_images) + [1] * len(nebulae_images) + [2] * len(star_images)
    # labels = np.array(labels)
    # labels = labels.reshape(-1, 1)
    # for i in range(10):
    #     plot_builder.display_image(images[i], str(labels[i]) + " - " + image_names[i])

    number_of_classes = len(set(labels))
    print(number_of_classes)
    print(Counter(labels))

    labels = to_categorical(labels, number_of_classes)

    # images = images / 255.0
    print(labels.shape)
    print(images.shape)
    MODEL_SAVE_NAME += str(len(images))

    data_train, data_test, labels_train, labels_test = train_test_split(images,
                                                                        labels,
                                                                        test_size=0.1,
                                                                        shuffle=True,
                                                                        random_state=1
                                                                        # shuffle=False,
                                                                        # random_state=None
                                                                        )
    del images, labels
    # for i in range(10):
    #     plot_builder.display_image(data_train[i], str(labels_train[i]) + " - " + image_names[i])
    validation_set_size = 100
    data_validate = data_train[-validation_set_size:]
    labels_validate = labels_train[-validation_set_size:]
    data_train = data_train[:-validation_set_size]
    labels_train = labels_train[:-validation_set_size]

    # reshape data for model compatibility
    data_train = data_train.reshape(-1, 1, number_of_pixels, number_of_pixels)
    data_test_orig = np.copy(data_test)
    data_validate_orig = np.copy(data_validate)
    data_test = data_test.reshape(-1, 1, number_of_pixels, number_of_pixels)
    data_validate = data_validate.reshape(-1, 1, number_of_pixels, number_of_pixels)

    # data_train = tf.expand_dims(data_train, axis=-1)
    # data_test = tf.expand_dims(data_test, axis=-1)
    # data_validate = tf.expand_dims(data_validate, axis=-1)

    # print(data_validate.shape, labels_validate.shape, data_train.shape, labels_train.shape, data_test.shape, labels_test.shape)

    train_model(data_train, data_test, labels_train, labels_test, data_validate, labels_validate, number_of_classes)

    evaluate(data_test_orig, np.where(labels_test == 1)[1], manual=True)
    evaluate(data_validate_orig, np.where(labels_validate == 1)[1], manual=True)
