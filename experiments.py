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

skips = [".jpg", " ",
"@", "+", "]", "[", ")", "(", "_",
"$", "z", "j", "b", "k", "v", "w", # less than 50
"A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O",
"P", "Q", "R","S", "T", "U", "V", "W", "X", "Y", "Z",
        ".", ",", "0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]

dataset, y, n_classes, label_dict, SIZE =  load_images_dataset.prepare_data(path, skips=skips)


print("Number of classes: {}".format(n_classes))

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

model = Sequential()
model.add(Conv2D(32,
                 kernel_size=(3, 3),
                 activation='tanh',
                 input_shape=SIZE))
#tanh offering more specific vals, rather than 1 0
model.add(Conv2D(64, (3, 3), activation='tanh')) # relu
print(x_train.shape)
model.add(keras.layers.ConvLSTM2D(32, (3,3), strides=(1, 1),
                        padding='valid',
                         dilation_rate=(1, 1), activation='tanh',
                         # data_format='channels_last',
                        # recurrent_activation='hard_sigmoid', use_bias=True,
                        # kernel_initializer='glorot_uniform',
                        # recurrent_initializer='orthogonal',
                        #  bias_initializer='zeros', unit_forget_bias=True,
                        #  kernel_regularizer=None, recurrent_regularizer=None,
                        #  bias_regularizer=None, activity_regularizer=None,
                        #  kernel_constraint=None, recurrent_constraint=None,
                        #   bias_constraint=None,
                        # return_sequences=True,
                        #    go_backwards=False, stateful=False, dropout=0.0,
                        #    recurrent_dropout=0.0))
                        ))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='tanh'))
model.add(Dropout(0.5))
model.add(Dense(n_classes, activation='sigmoid'))

 # 'categorical_crossentropy' <- supposedly for multi-class, not multi label: https://stats.stackexchange.com/questions/260505/machine-learning-should-i-use-a-categorical-cross-entropy-or-binary-cross-entro
model.compile(loss='binary_crossentropy',
#'binary_crossentropy' : supposedely ideal for multi label, current .5 test accuracy, but no letters predicted
# 'mean_squared_error' : all same, 1s
              optimizer=keras.optimizers.Adam(), #.Adam(), Adadelta()
              metrics=['categorical_accuracy', 'accuracy', 'mae'])

model.fit(x_train, y_train,
          batch_size=128,
          epochs=3,
          verbose=1,
          # validation_data=(x_test, y_test)
          validation_split=0.4
          )

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
        # print(index)
        actuals += " {}".format(label_dict["idx2word"][index])
    print("---------------------------------------\nActual: {}".format(actuals))

    # label_dict["idx2word"][s],y[n+i][s]) for s in y[n+i])
    # print("Prediction: {}".format(pred[i]))
    print("Predicted letters: ")
    for i2 in range (0, len(label_dict["idx2word"])):
        if pred[i][i2] > 0.0:
            print("\"{}\":{}".format(label_dict["idx2word"][i2], pred[i][i2]))
    print("--------------------------------------")
