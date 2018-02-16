
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


if os.name == 'nt':
    env = 0
else:
    env = 1

if env == 1:
    path="/home/ubuntu/Cursive-OCR-for-Geneology/dataset"
else:
    path="C:\\Users\\grant\\Repos\\Cursive-OCR-for-Geneology\\dataset"

# path = 'posters/'
# data = pd.read_csv("MovieGenre.csv", encoding="ISO-8859-1")
# data.head()


# image_glob = glob.glob(path + "/" + "*.jpg")
# img_dict = {}


# def get_id(filename):
#     index_s = filename.rfind("/") + 1
#     index_f = filename.rfind(".jpg")
#     return filename[index_s:index_f]

# for file in glob.glob("*.jpg"):
#     img_dict[file] = scipy.misc.imread(file)
#
# def show_img(id):
#     plt.imshow(img_dict[id])
#     plt.title("".format(str(id)))
#
# show_img("e (10).jpg")

# def preprocess(img, size=(150, 101)):
# def preprocess(img):
#     # img = scipy.misc.imresize(img, size)
#     img = img.astype(np.float32)
#     # img = (img / 127.5) - 1.
#     return img

# def prepare_data(data, img_dict, size=(150, 101)):
# def prepare_data(data, img_dict, size=(60, 40)):
#     print("Generation dataset...")
#     dataset = []
#     y = []
#     ids = []
    # label_dict = {"word2idx": {}, "idx2word": []}
#     idx = 0
#
#     # here, I think, classes are specified
#     # genre_per_movie = data["Genre"].apply(lambda x: str(x).split("|"))
#     letters_per_window = data["letters"].apply(lambda x: str(x).split("|"))
#
#     for l in [g for d in letters_per_window for g in d]:
#         if l in label_dict["idx2word"]:
#             pass
#         else:
#             label_dict["idx2word"].append(l)
#             label_dict["word2idx"][l] = idx
#             idx += 1
#     n_classes = len(label_dict["idx2word"])
#     print("identified {} classes".format(n_classes))
#     n_samples = len(img_dict)
#     print("got {} samples".format(n_samples))
#     for k in img_dict:
#         try:
#             g = data[data["imdbId"] == int(k)]["Genre"].values[0].split("|")
#             img = preprocess(img_dict[k], size)
#             if img.shape != (60, 40, 3):
#                 # if img.shape != (150, 101, 3):
#                 continue
#             # l = sum of classes and labels)
#             l = np.sum([np.eye(n_classes, dtype="uint8")[label_dict["word2idx"][s]]
#                                                         for s in g], axis=0)
#             y.append(l)
#             dataset.append(img)
#             ids.append(k)
#         except:
#             pass
#     print("DONE")
#     return dataset, y, label_dict, ids

# my way
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
        print("letters: {}\nlabel: {}".format(letters, l))
        y.append(l)

    return imgs, y

SIZE = (60, 40)
# dataset = imgs, label_dict = word2indx:, indx2word:, ids = img titles, y = list of sumed classes and label indexes
# dataset, y, label_dict, ids =  prepare_data(data, img_dict, size=SIZE)
dataset, y =  prepare_data(path)

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
model.add(Dense(29, activation='sigmoid'))

model.compile(loss='binary_crossentropy',
              optimizer=keras.optimizers.Adam(),
              metrics=['accuracy'])

n = 10000
model.fit(np.array(dataset[: n]), np.array(y[: n]), batch_size=16, epochs=5,
          verbose=1, validation_split=0.1)

n_test = 100
X_test = dataset[n:n + n_test]
# y_test = y[n:n + n_test]

pred = model.predict(np.array(X_test))

def show_example(idx):
    # N_true = int(np.sum(y_test[idx]))
    show_img(ids[n + idx])
    # print("Prediction: {}".format("|".join(["{} ({:.3})".format(label_dict["idx2word"][s],pred[idx][s])
                                                        # for s in pred[idx].argsort()[-N_true:][::-1]])))

show_example(3)
