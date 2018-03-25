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

path = os.getcwd() + "/dataset"

# n_classes = 2
base_layers = 3
epochs = 10
batch_size = 10
conv_size = 3
pool_size = 2

train_imgs, train_labels, val_imgs, val_labels, name_labels, n_classes, input_shape = \
        load_images_dataset.read_my_csv("train_has_e.txt", "val_has_e.txt", \
        input_shape=(60, 70, 3), channels=3, one_hot=False)

x_train, x_val, x_test, y_train, y_val, y_test, n_test, n = \
load_images_dataset.divide_data_with_val(train_imgs, train_labels, val_imgs, val_labels)
# n_test, n, x_test, x_train, y_test, y_train, train_name_labels, test_name_labels  = \
# n_test, n, x_test, x_train, y_test, y_train = load_images_dataset.divide_data(imgs, labels, name_labels)

# print(x_train[:5])

# for x in range(1, 10):
#     print(labels[x], name_labels[x])

model = custom_models.basic_cnn('relu', 'mean_squared_error', \
                                x_train, y_train, x_val, y_val, input_shape, n_classes, \
                                epochs=epochs, batch_size=batch_size)

score = model.evaluate(x_test, y_test, verbose=1)

print('Test loss:', score[0])
print('Test accuracy:', score[1])
print(score)

pred = model.predict(x_test)

print("predictions finished")

for i in range (0, len(x_test)):
    actuals = ""
    # for label in y[n+i]:
    for index in np.where(val_labels[n+i]==1)[0]:
        # actuals += " {}".format(label_dict["idx2word"][index])
        actuals += str(index+1)
    print("---------------------------------------\nActual: {}".format(actuals))

    # label_dict["idx2word"][s],y[n+i][s]) for s in y[n+i])
    # print("Prediction: {}".format(pred[i]))

    preds = pred[i]
    formatted_preds = []
    for ind, val in enumerate(preds):
        # print(ind)
        # print(val)
        formatted_preds.append("{} probability of label: {}".format(val, ind+1))
    formatted_preds.sort()
    for x in formatted_preds:
        print(x)
    # print("Predicted: {}".format(formatted_preds.sort()))
    # for i2 in range (0, len(label_dict["idx2word"])):
        # if pred[i][i2] > 0.2:
        # print("\"{}\":{}".format(label_dict["idx2word"][i2], pred[i][i2]))
    print("--------------------------------------")
