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


def prepare_data(imgs_dir,
                SIZE=(60,70,3),
                skips=[".jpg", " "]):
    label_dict = {"word2idx": {}, "idx2word": []}
    os.chdir(imgs_dir)
    labels = []
    idx = 0

    imgs = []
    clean_titles = []
    label_cardinality = {}
    for file in glob.glob("*/*.jpg", recursive=True):

        img = scipy.misc.imread(file).astype(np.float32)

        if img.shape[0] == SIZE[0] and img.shape[1] == SIZE[1] and img.shape[2] == SIZE[2]:

            clean_title = str(file.split('\\')[1])
            clean_title = re.sub(r"\([\d+]*\)", "", clean_title)

            for lb in skips:

                clean_title = clean_title.replace(lb, "")

            if len(clean_title) > 0:

                imgs.append(img)
                clean_titles.append(clean_title)
        else:
            print("{} size mismatch: {}".format(file, img.shape))


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
        labels.append(l)

    # print(label_cardinality)
    for l in sorted(label_cardinality):
        print(l, ": ", label_cardinality[l])

    return imgs, labels, n_classes, label_dict, SIZE


def read_my_csv(file_name, input_shape=(60, 70, 3), delimiter='/', channels=3, one_hot=True):
    """
    This function is used to pull specific label atrributes from a file,
    in addition to processing the input images.
    """
    print("reading data from csv: {} with delimiter '{}'".format(file_name, delimiter))

    path = os.getcwd() + "\\{}".format(file_name)
    df = pd.read_csv(path, delimiter=delimiter)

    imgs = []
    labels = []
    names = []
    flatten = True if channels == 2 else False

    n_classes = df.apply(pd.Series.nunique)['Y']

    # eyes = np.eye(n_classes, dtype="uint8")
    # print(expected_shape, delimiter)
    for index, row in df.iterrows():
        try:
            img = scipy.misc.imread(row.X, flatten=flatten).astype(np.float32)

            assert img.shape == input_shape
            imgs.append(img)
            if one_hot:
                labels.append(eyes[(row.Y)-1])
            else:
                labels.append(row.Y)
            names.append(row.X)
        except Exception as e:
            print("Failed img error: {} : {}".format(row.X, e))

    print("Total dataset: {}".format(len(imgs)))
    print("Total labels: {}".format(len(labels)))


    assert len(imgs) == len(labels)

    return imgs, labels, names, n_classes, input_shape


def divide_data(imgs, labels, name_labels, n_test=10):
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

    return n_test, n, x_test, x_train, y_test, y_train
