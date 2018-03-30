import os
import numpy as np
import pandas as pd
import glob
import scipy.misc
import matplotlib
# %matplotlib inline
import matplotlib.pyplot as plt
import string
import re
import random
import custom_models
import cv2
import load_images_dataset



x, y, n_classes, label_dict, size = load_images_dataset.prepare_data(
        'C:\\Users\\grant\\Repos\\Cursive-OCR-for-Geneology\\dataset\\ready_singles')

n_test, n_classes, x_test, x_train, y_test, y_train = load_images_dataset.divide_data(
        x, y, label_dict, n_test=10)


epochs = 10
batch_size = 24


model = custom_models.basic_cnn('relu', 'softmax', \
                        x_train, y_train, size, 5, \
                        epochs=epochs, batch_size=batch_size)

score = model.evaluate(x_test, y_test, verbose=1)

print('Test loss:', score[0])
print('Test accuracy:', score[1])
print(score)

pred = model.predict(x_test)

print("predictions finished")
print(pred)
print(y_test)
