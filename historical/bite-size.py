"""
The goal of this file is to establish smaller classification methods which can be used as inputs to a more intelligent model

Attempt to segment letters into general categories to help with classification
Use attributes like:
    Contains tall letters or no
    Contains numbers
    Is single letter or multiple letters
    Has space or no (two word parts or all one)
    Has Capital letter or no

"""
import numpy as np
import random
import os
from keras.preprocessing.image import ImageDataGenerator

import load_images_dataset
import custom_models

from keras.datasets import mnist


path = os.getcwd() + "/dataset/single"

# n_classes = 2
# base_layers = 3
epochs = 20
batch_size = 24
# conv_size = 3
# pool_size = 2

imgs, labels, n_classes, label_dict, SIZE = load_images_dataset.prepare_data(path,
                                                            SIZE=(60,25),
                                                            skips=[".jpg", " "])

 # train_name_labels, test_name_labels
n_test, n, x_test, x_train, y_test, y_train = \
                                load_images_dataset.divide_data(imgs, labels, label_dict, n_test=10)

# train_imgs, train_labels, val_imgs, val_labels, name_labels, n_classes, input_shape = \
#         load_images_dataset.read_my_csv("train_sameheight.txt", "val_sameheight.txt", \
#         input_shape=(60, 70), channels=2, one_hot=True)



# x_train, x_val, x_test, y_train, y_val, y_test, n_test, n = \
# load_images_dataset.divide_data_with_val(train_imgs, train_labels, val_imgs, val_labels)
"""Make my a vs e dataset so big that it is like mnist (70K total, about 8K per class)"""

# -----------MNIST DATA TEST------------
# (x_train, y_train), (x_test, y_test) = mnist.load_data()
#
# eyes=np.eye(10, dtype="uint8")
#
# new_y_train = []
# new_y_test = []
#
# for i, v in enumerate(y_train):
#     new_y_train.append(eyes[v])
# for i, v in enumerate(y_test):
#     new_y_test.append(eyes[v])
#
# new_y_test = np.array(new_y_test)
# new_y_train = np.array(new_y_train)



model = custom_models.basic_cnn('relu', 'mean_squared_error', \
                        x_train, y_train, (60, 25), 2, \
                        epochs=epochs, batch_size=batch_size)


# n_test, n, x_test, x_train, y_test, y_train, train_name_labels, test_name_labels  = \
# n_test, n, x_test, x_train, y_test, y_train = load_images_dataset.divide_data(imgs, labels, name_labels)

# print(x_train[:5])

# for x in range(1, 10):
#     print(labels[x], name_labels[x])

# model = custom_models.basic_cnn('relu', 'mean_squared_error', \
#                                 x_train, y_train, x_val, y_val, input_shape, n_classes, \
#                                 epochs=epochs, batch_size=batch_size)

score = model.evaluate(x_test, y_test, verbose=1)

print('Test loss:', score[0])
print('Test accuracy:', score[1])
print(score)

pred = model.predict(x_test)

print("predictions finished")
print(pred)
print(y_test)
#
# for i in range (0, len(x_test)):
#     actuals = ""
#     # for label in y[n+i]:
#     for index in np.where(val_labels[n+i]==1)[0]:
#         # actuals += " {}".format(label_dict["idx2word"][index])
#         actuals += str(index+1)
#     print("---------------------------------------\nActual: {}".format(actuals))
#
#     # label_dict["idx2word"][s],y[n+i][s]) for s in y[n+i])
#     # print("Prediction: {}".format(pred[i]))
#
#     preds = pred[i]
#     formatted_preds = []
#     for ind, val in enumerate(preds):
#         # print(ind)
#         # print(val)
#         formatted_preds.append("{} probability of label: {}".format(val, ind+1))
#     formatted_preds.sort()
#     for x in formatted_preds:
#         print(x)
#     # print("Predicted: {}".format(formatted_preds.sort()))
#     # for i2 in range (0, len(label_dict["idx2word"])):
#         # if pred[i][i2] > 0.2:
#         # print("\"{}\":{}".format(label_dict["idx2word"][i2], pred[i][i2]))
#     print("--------------------------------------")
