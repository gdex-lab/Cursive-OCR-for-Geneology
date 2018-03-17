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



# n_classes = 2
base_layers = 3
epochs = 45
batch_size = 12
conv_size = 4
pool_size = 2


# dataset, y, name_labels, n_name_classes, name_label_dict, input_shape = \
#             load_images_dataset.read_my_csv("n_letters.txt", 8, '/')
# model = custom_models.seven_layer_cnn('relu', 'softmax', 'categorical_crossentropy', \
#                                         x_train, y_train, input_shape, 8, 3)

imgs, labels, name_labels, n_classes, input_shape = \
        load_images_dataset.read_my_csv("has_tall_letters_undersampled.txt", \
        input_shape=(60, 70), channels=2)

def divide_data(imgs, labels, name_labels, n_test=10, specific_validation=False, pct_valdation=0.4):
    # to unorder samples
    random_seed = 35
    random.Random(random_seed).shuffle(labels)
    random.Random(random_seed).shuffle(name_labels)
    random.Random(random_seed).shuffle(imgs)
    n = len(imgs) - (1+n_test)

    x_train = np.array(imgs[: n])
    y_train = np.array(labels[: n])

    x_test = np.array(imgs[n:n + n_test])
    y_test = np.array(labels[n:n + n_test])
    #
    # def oversample(x_train, y_train):
    #     from imblearn.over_sampling import SMOTE, ADASYN
    #     X_resampled, y_resampled = SMOTE().fit_sample(x_train, y_train)
    #     print("Resampling...")
    #     print(sorted(Counter(y_resampled).items()))
    #     #
    #     # clf_smote = LinearSVC().fit(X_resampled, y_resampled)
    #     # X_resampled, y_resampled = ADASYN().fit_sample(X, y)
    #     # print(sorted(Counter(y_resampled).items()))
    #     # clf_adasyn = LinearSVC().fit(X_resampled, y_resampled)
    #     return X_resampled, y_resampled
    #
    # if specific_validation:
    #     slice = int(n*pct_valdation)
    #     x_validation = x_train[:slice]
    #     y_validation = y_train[:slice]
    #     x_train, y_train = oversample(x_train[slice:], y_train[slice:])
    #     return n_test, n, x_test, x_train, y_test, y_train, x_validation, y_validation
    # else:
    return n_test, n, x_test, x_train, y_test, y_train



n_test, n, x_test, x_train, y_test, y_train = divide_data(imgs, labels, name_labels)
# print(x_train[:5])
for x in range(1, 10):
    print(labels[x], name_labels[x])

model = custom_models.basic_cnn('relu', 'mean_squared_error', \
                                x_train, y_train, input_shape, n_classes, \
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
    for index in np.where(labels[n+i]==1)[0]:
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
