from keras.preprocessing.image import ImageDataGenerator
import os
from matplotlib import pyplot
from keras import backend as K
import random
import numpy as np
import load_images_dataset
import custom_models
from keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
from keras.utils import to_categorical
from matplotlib import pyplot
import cv2
from numpy import array


x, y, n_classes, label_dict, size = load_images_dataset.prepare_data(
        'C:\\Users\\grant\\Repos\\Cursive-OCR-for-Geneology\\dataset\\ready_singles')

print(label_dict)
K.set_image_dim_ordering('tf')
x = array(x)
x.reshape(60, 25, 3, x.shape[0])
x = x.astype('float32')

# for z in range(0,2):
#     pyplt.imshow(np.uint8(x[z]))
#     pyplt.show()

# define data preparation
datagen = ImageDataGenerator(
    # featurewise_center=False,
    # samplewise_center=False,
    # featurewise_std_normalization=False,
    # samplewise_std_normalization=False,
    # zca_whitening=True,
    # zca_epsilon=1e-6,
    rotation_range=12,
    width_shift_range=4,
    height_shift_range=10,
    # shear_range=0.,
    zoom_range=0.2,
    channel_shift_range=.8,
    # fill_mode='nearest',
    # cval=0.,
    horizontal_flip=False,
    vertical_flip=False,
    rescale=0,
    # preprocessing_function=None,
    data_format=K.image_data_format())

# print(label_dict.shape)
print('fitting augmentation model')
datagen.fit(x)
aug_cnt = 0

# number of times to augment the dataset
batch_size = 1
aug_factor = len(x)/batch_size * 20


print('creating flow')
path = 'C:\\Users\\grant\\Repos\\Cursive-OCR-for-Geneology\\dataset\\01a_singles_augmented'
for x_batch, y_batch in datagen.flow(x, y, batch_size=batch_size):
    try:
        label = '{}\\{} ({}).jpg'.format(
                    path,
                    label_dict['idx2word'][int(np.where(y_batch[0]==1)[0][0])],
                    aug_cnt
                    )
        # print(label)
        """
        pyplot has a hard time showing major changes, but you will still get an idea for the variety of the pixels.
            Output the images to actually see appearance.
        """
        # pyplot.imshow(x_batch[0])
        # pyplot.show()
        aug_cnt += 1

        cv2.imwrite(label, x_batch[0])
        # print('writing {}\\{}{}.jpg'.format(path,os.path.split(y_batch[0])[1].strip('.jpg'), '({})'.format(aug_cnt)))
        # cv2.imwrite('{}\\{}{}.jpg'.format(path,os.path.split(y_batch[0])[1].strip('.jpg'), '({})'.format(random.getrandbits(8))),
                        # X_batch.reshape(60, 70, 3))
        # print('wrote')
    except Exception as e:
        print(e)
    if aug_cnt > aug_factor:
        # flow will loop indefinitely
        break
