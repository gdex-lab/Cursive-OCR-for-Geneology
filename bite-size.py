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

import load_images_dataset
import custom_models

path = os.getcwd() + "/dataset"
n_classes = 8

dataset, y, name_labels, n_name_classes, name_label_dict, input_shape = \
            load_images_dataset.read_my_csv("n_letters.txt", '/')

# to unorder samples
random_seed = 4
random.Random(random_seed).shuffle(y)
random.Random(random_seed).shuffle(dataset)

n_test = 8
n = len(dataset) -(1+n_test)
x_test = np.array(dataset[n:n + n_test])
x_train = np.array(dataset[: n])
y_test = np.array(y[n:n + n_test])
y_train = np.array(y[: n])

# model = custom_models.nine_layer_cnn('tanh', 'sigmoid', 'binary_crossentropy', \
#                                         x_train, y_train, input_shape, n_classes)

model = custom_models.nine_layer_cnn('relu', 'softmax', 'categorical_crossentropy', \
                                        x_train, y_train, input_shape, n_classes, 20)

score = model.evaluate(x_test, y_test, verbose=1)

print('Test loss:', score[0])
print('Test accuracy:', score[1])
print(score)

pred = model.predict(x_test)

print("predictions finished")


for i in range (0, len(x_test)):
    actuals = ""
    # for label in y[n+i]:
    for index in np.where(y[n+i]==1)[0]:
        # actuals += " {}".format(label_dict["idx2word"][index])
        actuals += str(index+1)
    print("---------------------------------------\nActual: {}".format(actuals))

    # label_dict["idx2word"][s],y[n+i][s]) for s in y[n+i])
    # print("Prediction: {}".format(pred[i]))
    print("Predicted: {}".format(pred[i]))
    # for i2 in range (0, len(label_dict["idx2word"])):
        # if pred[i][i2] > 0.2:
        # print("\"{}\":{}".format(label_dict["idx2word"][i2], pred[i][i2]))
    print("--------------------------------------")
