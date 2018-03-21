from keras.preprocessing.image import ImageDataGenerator
from matplotlib import pyplot
import os
from keras import backend as K
import random
import numpy as np
import load_images_dataset
import custom_models

imgs, labels, name_labels, n_classes, input_shape = \
load_images_dataset.read_my_csv("has_tall_letters_undersampled.txt", \
input_shape=(60, 70, 3), channels=3, one_hot=False)


n_test, n, X_test, X_train, y_test, y_train = load_images_dataset.divide_data(imgs, labels, name_labels)

# (featurewise_center=False,
#     samplewise_center=False,
#     featurewise_std_normalization=False,
#     samplewise_std_normalization=False,
#     zca_whitening=False,
#     zca_epsilon=1e-6,
#     rotation_range=0.,
#     width_shift_range=0.,
#     height_shift_range=0.,
#     shear_range=0.,
#     zoom_range=0.,
#     channel_shift_range=0.,
#     fill_mode='nearest',
#     cval=0.,
#     horizontal_flip=False,
#     vertical_flip=False,
#     rescale=None,
#     preprocessing_function=None,
#     data_format=K.image_data_format())


from keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
from matplotlib import pyplot
import os
from keras import backend as K
K.set_image_dim_ordering('th')
# load data
# (X_train, y_train), (X_test, y_test) = mnist.load_data()


# reshape to be [samples][pixels][width][height]
X_train = X_train.reshape(X_train.shape[0], 3, 60, 70)
X_test = X_test.reshape(X_test.shape[0], 3, 60, 70)
# convert from int to float
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')

import matplotlib.pyplot as plt
for z in range(0,10):
    plt.imshow(np.uint8(imgs[z]))
    plt.show()

# define data preparation
datagen = ImageDataGenerator(
rotation_range=10,
width_shift_range=0.1,
height_shift_range=0.1,
shear_range=0.1,
zoom_range=0.1,
horizontal_flip=True,
fill_mode='nearest')
# fit parameters from data
datagen.fit(X_train)
# configure batch size and retrieve one batch of images
# os.makedirs('.\\dataset\\augmented')
for X_batch, y_batch in datagen.flow(X_train, y_train, batch_size=9,
save_to_dir='.\\dataset\\augmented', save_prefix='aug', save_format='jpeg'):
# create a grid of 3x3 images
    for i in range(0, 9):
        pyplot.subplot(330 + 1 + i)
        pyplot.imshow(X_batch[i].reshape(60, 70, 3))
        print(y_batch[i])
    # show the plot
    pyplot.show()
    break
