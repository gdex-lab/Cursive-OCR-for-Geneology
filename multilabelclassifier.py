
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

label_dict = {"label2idx": {},
            "idx2label": []}

classes = []
non_letters = ["space", "noise", "partial", "$", "&", "open_paren", "close_paren"]
for cL in list(string.ascii_lowercase):
    classes.append(cL)
for cU in list(string.ascii_uppercase):
    classes.append(cU)
for number in range(0, 10):
    classes.append(number)
for other in non_letters:
    classes.append(other)

label_dict = {"word2idx": {}, "idx2word": []}
def prepare_data(imgs_dir):
    os.chdir(imgs_dir)
    y = []
    idx = 0

    imgs = []
    clean_titles = []
    for file in glob.glob("*.jpg"):
        imgs.append(scipy.misc.imread(file).astype(np.float32))
        clean_titles.append(re.sub(r"\([\d+]*\)", "", str(file.replace(".jpg", "").replace(" ", "")).rstrip()))

    # Add all file labels to dict, with indexes
    for title in clean_titles:
        for l in title.split('|'):
            if l in label_dict["idx2word"]:
                pass
            else:
                label_dict["idx2word"].append(l)
                label_dict["word2idx"][l] = idx
                idx += 1

    print(label_dict)

    # add multi-hot labels to overall labels?
    for title in clean_titles:
        letters = title.split("|")
        n_classes = len(label_dict["idx2word"])
        l = np.sum([np.eye(n_classes, dtype="uint8")[label_dict["word2idx"][s]]
                                                            for s in letters], axis=0)
        # print("letters: {}\nlabel: {}".format(letters, l))
        y.append(l)

    return imgs, y

SIZE = (60, 40)
# dataset = imgs, label_dict = word2indx:, indx2word:, ids = img titles, y = list of sumed classes and label indexes
# dataset, y, label_dict, ids =  prepare_data(data, img_dict, size=SIZE)
dataset, y =  prepare_data(path)


print("shuffling dataset")
# to unorder samples
random.Random(4).shuffle(y)
random.Random(4).shuffle(dataset)

# rand = np.random.RandomState(5)

# shuffle = rand.permutation(500) # len of windows turns out larger than 84?? 252??
# print
# print(rand, '\n\n\n',len(windows),'\n\n\n', shuffle)
# dataset, y = dataset[shuffle], y[shuffle]

# print(dataset[0].shape)

# ------------------------------------

import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization

model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3), activation='relu',
                 input_shape=(SIZE[0], SIZE[1], 3)))

model.add(Conv2D(32, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(9, activation='sigmoid')) # was 29 instead of 9

model.compile(loss='binary_crossentropy',
              optimizer=keras.optimizers.Adam(),
              metrics=['accuracy'])

# current 547 train and vaildate on 500, test examples for predictions on 47
# total imgs?
n_test = 30
n = len(dataset) -(1+n_test)

print("Beginning fit...")
model.fit(np.array(dataset[: n]), np.array(y[: n]), batch_size=4, epochs=300,
          verbose=1, validation_split=0.1)

X_test = dataset[n:n + n_test]
y_test = y[n:n + n_test]

print(len(X_test))
print("Ytest: ",len(y_test))

print("model.fit DONE. Moving on to pred...")
pred = model.predict(np.array(X_test))
print("predictions finished")

for i in range (0, len(X_test)):
    print("---------------------------------------\nActual: {}".format(label_dict["idx2word"][np.where(y[n+i]==1)[0][0]]))
    print("Prediction: {}".format(pred[i]))
    print("Details...")
    for i2 in range (0, len(label_dict["idx2word"])):
        print("\"{}\" probability:{}".format(label_dict["idx2word"][i2], pred[i][i2]))
        print("--------------------------------------")
#
# def show_images(images, cols = 1, titles = None):
#     """Display a list of images in a single figure with matplotlib.
#
#     Parameters
#     ---------
#     images: List of np.arrays compatible with plt.imshow.
#
#     cols (Default = 1): Number of columns in figure (number of rows is
#                         set to np.ceil(n_images/float(cols))).
#
#     titles: List of titles corresponding to each image. Must have
#             the same length as titles.
#     """
#     assert((titles is None)or (len(images) == len(titles)))
#     n_images = len(images)
#     if titles is None: titles = ['Image (%d)' % i for i in range(1,n_images + 1)]
#     fig = plt.figure()
#     for n, (image, title) in enumerate(zip(images, titles)):
#         a = fig.add_subplot(cols, np.ceil(n_images/float(cols)), n + 1)
#         if image.ndim == 2:
#             plt.gray()
#         plt.imshow(image)
#         a.set_title(title)
#     fig.set_size_inches(np.array(fig.get_size_inches()) * n_images)
#     mng = plt.get_current_fig_manager()
#     # mng.window.state('zoomed')
#     plt.show()



# def show_example(idx):
#     N_true = int(np.sum(y_test[idx]))
#     print("Actual: {}".format(label_dict["idx2word"][np.where(y[n+idx]==1)[0][0]]))
#     print("Prediction: {}".format("|".join(["{} ({:.3})".format(label_dict["idx2word"][s],pred[idx][s])
#                                 for s in pred[idx].argsort()[-N_true:][::-1]])))
#     show_images([X_test[idx]])
# plt.imshow(dataset[0])

# show_example(0)
# show_example(1)
# show_example(2)
# show_example(3)
# show_example(4)
# show_example(5)
# show_example(6)
# show_example(7)
