from __future__ import print_function
import cv2
import numpy as np
from matplotlib import pyplot as plt

import numpy as np
import pandas as pd
import scipy.misc
import matplotlib
# %matplotlib inline
import matplotlib.pyplot as plt
import string
import re
import random
import os


if os.name == 'nt':
    env = 0
else:
    env = 1

if env == 1:
    path="/home/ubuntu/Cursive-OCR-for-Geneology/dataset"
else:
    path="C:\\Users\\grant\\Repos\\Cursive-OCR-for-Geneology\\dataset"

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
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
import load_images_dataset


dataset, y, n_classes, label_dict, SIZE =  load_images_dataset.prepare_data(path)


print("shuffling dataset")
print("Number of classes: {}".format(n_classes))
# to unorder samples
random_seed = 4
random.Random(random_seed).shuffle(y)
random.Random(random_seed).shuffle(dataset)

n_test = 8
n = len(dataset) -(1+n_test)
x_test = np.array(dataset[n:n + n_test])
y_test = y[n:n + n_test]
x_train = np.array(dataset[: n])
y_train = np.array(y[: n])

model = Sequential()
model.add(Conv2D(32,
                 kernel_size=(3, 3),
                 activation='relu',
                 input_shape=SIZE))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(n_classes, activation='sigmoid')) #softmax

model.compile(loss=keras.losses.categorical_crossentropy, #'binary_crossentropy', 'categorical_crossentropy'
              optimizer=keras.optimizers.Adam(), #.Adam(), Adadelta()
              metrics=['accuracy', 'mae'])

model.fit(x_train, y_train,
          batch_size=128,
          epochs=3,
          verbose=1,
          # validation_data=(x_test, y_test)
          )
score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
print(score)
