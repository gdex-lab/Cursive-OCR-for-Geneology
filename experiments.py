from __future__ import print_function
import cv2
import numpy as np
from matplotlib import pyplot as plt

import numpy as np
import pandas as pd
import glob
import scipy.misc
import matplotlib
# %matplotlib inline
import matplotlib.pyplot as plt
import string
import os
import re
import random


if os.name == 'nt':
    env = 0
else:
    env = 1

if env == 1:
    path="/home/ubuntu/Cursive-OCR-for-Geneology/dataset"
else:
    path="C:\\Users\\grant\\Repos\\Cursive-OCR-for-Geneology\\dataset"
#
os.chdir(path)
# for file in glob.glob("*/*.jpg", recursive=True):
#     img = cv2.imread(file,0)
#     edges = cv2.Canny(img,100,200)

    # plt.subplot(121),plt.imshow(img,cmap = 'gray')
    # plt.title('Original Image'), plt.xticks([]), plt.yticks([])
    # plt.subplot(122),plt.imshow(edges,cmap = 'gray')
    # plt.title('Edge Image'), plt.xticks([]), plt.yticks([])

    # plt.show()

'''Trains a simple convnet on the MNIST dataset.
Gets to 99.25% test accuracy after 12 epochs
(there is still a lot of margin for parameter tuning).
16 seconds per epoch on a GRID K520 GPU.
'''

import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K

batch_size = 128
num_classes = 10
epochs = 12

# input image dimensions

SIZE = (60, 70)
img_rows, img_cols = SIZE[0], SIZE[1]



label_dict = {"word2idx": {}, "idx2word": []}
def prepare_data(imgs_dir):
    os.chdir(imgs_dir)
    y = []
    idx = 0
    skips = [".jpg", " ", "@", "+", "]", "[", ")", "(", "_",
    "$", "z", "j", "b", "k", "v", "w", # less than 50
    "A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O",
    "P", "Q", "R","S", "T", "U", "V", "W", "X", "Y", "Z",
            ".", ",", "0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]
    imgs = []
    clean_titles = []
    label_cardinality = {}
    for file in glob.glob("*/*.jpg", recursive=True):
        img = scipy.misc.imread(file).astype(np.float32)

        if img.shape[0] == SIZE[0] and img.shape[1] == SIZE[1] and img.shape[2] == 3:
            clean_title = str(file.split('\\')[1])
            clean_title = re.sub(r"\([\d+]*\)", "", clean_title)
            for lb in skips:
                clean_title = clean_title.replace(lb, "")

            if len(clean_title) > 0:
                imgs.append(img)
                # print(clean_title)
                clean_titles.append(clean_title)
        else:
            print("img size mismatch: {}".format(img.shape))


    # Add all file labels to dict, with indexes
    for title in clean_titles:
        for l in list(title): #.split('|'):
            if l in label_cardinality:
                label_cardinality[l] += 1
            else:
                label_cardinality[l] = 1
            if l in label_dict["idx2word"]:
                pass
            else:
                label_dict["idx2word"].append(l)
                label_dict["word2idx"][l] = idx
                idx += 1


    n_classes = len(label_dict["idx2word"])
    # add multi-hot labels to overall labels?
    for title in clean_titles:
        letters = list(title)
        l = np.sum([np.eye(n_classes, dtype="uint8")[label_dict["word2idx"][s]]
                                                            for s in letters], axis=0)
        # print("letters: {}\nlabel: {}".format(letters, l))
        y.append(l)

    # print(label_cardinality)
    for l in sorted(label_cardinality):
        print(l, ": ", label_cardinality[l])

    return imgs, y, n_classes

# dataset = imgs, label_dict = word2indx:, indx2word:, ids = img titles, y = list of sumed classes and label indexes
# dataset, y, label_dict, ids =  prepare_data(data, img_dict, size=SIZE)
dataset, y, n_classes =  prepare_data(path)


print("shuffling dataset")
print("Number of classes: {}".format(n_classes))
# to unorder samples
random_seed = 4
random.Random(random_seed).shuffle(y)
random.Random(random_seed).shuffle(dataset)

n_test = 8
n = len(dataset) -(1+n_test)
x_test = dataset[n:n + n_test]
y_test = y[n:n + n_test]
x_train = dataset[: n]
y_train = np.array(y[: n])


# the data, split between train and test sets
# (x_train, y_train), (x_test, y_test) = mnist.load_data()
# x_train =






#
# if K.image_data_format() == 'channels_first':
#     x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
#     x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
#     input_shape = (1, img_rows, img_cols)
# else:
#     x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
#     x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
#     input_shape = (img_rows, img_cols, 1)
#
# x_train = x_train.astype('float32')
# x_test = x_test.astype('float32')
# x_train /= 255
# x_test /= 255
# print('x_train shape:', x_train.shape)
# print(x_train.shape[0], 'train samples')
# print(x_test.shape[0], 'test samples')

# convert class vectors to binary class matrices
# y_train = keras.utils.to_categorical(y_train, n_c/lasses)
# y_test = keras.utils.to_categorical(y_test, num_classes)

model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=(img_rows, img_cols, 1)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='sigmoid'))

model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])

model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_data=(x_test, y_test))
score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
