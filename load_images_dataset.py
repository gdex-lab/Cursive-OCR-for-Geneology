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


def read_my_csv(train_file_name, val_file_name, input_shape=(60, 70, 3), delimiter='/', channels=3, one_hot=True):
    """
    This function is used to pull specific label atrributes from a file,
    in addition to processing the input images.
    """
    print("Reading training data from csv: {} with delimiter '{}'".format(train_file_name, delimiter))

    # path = os.getcwd() + "\\{}".format(file_name)
    train_df = pd.read_csv(train_file_name, delimiter=delimiter)

    print("Reading validation data from csv: {} with delimiter '{}'".format(val_file_name, delimiter))
    val_df = pd.read_csv(val_file_name, delimiter=delimiter)

    train_imgs = []
    train_labels = []
    val_imgs = []
    val_labels = []
    names = []
    flatten = True if channels == 2 else False

    print("Detecting distinct classes")
    n_classes = train_df.apply(pd.Series.nunique)['Y']
    val_n_classes = val_df.apply(pd.Series.nunique)['Y']
    assert n_classes == val_n_classes, "Number of classes in training and validation data mismatch"
    eyes = np.eye(n_classes, dtype="uint8")

    kernel = np.ones((5,9),np.uint8) # was 5, 5, first is vert, then horizontal
    # print(expected_shape, delimiter)
    print("Iterating training rows")
    print("-Reading images")
    print("-Dilating images to repair errosion")
    print("-Storing labels")
    for index, row in train_df.iterrows():
        try:
            name = row.X
            img = scipy.misc.imread(name, flatten=flatten).astype(np.float32)
            # img = cv2.imread(name)
            # img = cv2.erode(img,kernel,iterations = 1)
            # img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            assert img.shape == input_shape, "Image shape {} does not match input shape {}".format(img.shape, input_shape)
            train_imgs.append(img)

            # names.append(name)
            if one_hot:
                train_labels.append(eyes[(row.Y)-1])
            else:
                train_labels.append(row.Y)
        except Exception as e:
            print("Failed img error: {} : {}".format(name, e))

    print("Iterating validation rows")
    print("-Reading images")
    print("-Dilating images to repair errosion")
    print("-Storing labels")
    for index, row in val_df.iterrows():
        try:
            name = row.X
            img = scipy.misc.imread(name, flatten=flatten).astype(np.float32)
            # img = cv2.imread(name)
            # img = cv2.erode(img,kernel,iterations = 1)
            # img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            assert img.shape == input_shape, "Image shape {} does not match input shape {}".format(img.shape, input_shape)
            val_imgs.append(img)

            # names.append(name)
            if one_hot:
                val_labels.append(eyes[(row.Y)-1])
            else:
                val_labels.append(row.Y)
        except Exception as e:
            print("Failed img error: {} : {}".format(name, e))

    print("Total training imgs: {}".format(len(train_imgs)))
    print("Total validation imgs: {}".format(len(val_imgs)))
    print("Total training labels: {}".format(len(train_labels)))
    print("Total validation labels: {}".format(len(val_labels)))
    # print("Total names: {}".format(len(names)))

    assert len(train_imgs) == len(train_labels), "Training images mismatch with training labels"
    assert len(val_imgs) == len(val_labels), "Validation images mismatch with validation labels"

    return train_imgs, train_labels, val_imgs, val_labels, names, n_classes, input_shape

def divide_data_with_val(train_imgs, train_labels, val_imgs, val_labels, n_test=10):
    # to unorder samples
    random_seed = 35
    random.Random(random_seed).shuffle(train_imgs)
    random.Random(random_seed).shuffle(train_labels)
    # random.Random(random_seed).shuffle(train_name_labels)
    random.Random(random_seed).shuffle(val_imgs)
    random.Random(random_seed).shuffle(val_labels)
    # random.Random(random_seed).shuffle(val_name_labels)
    n = len(val_imgs) - (1+n_test)

    x_val = np.array(val_imgs[:n])
    y_val = np.array(val_labels[:n])
    # train_name_labels = np.array(val_name_labels[: n])

    x_train = np.array(train_imgs)
    y_train = np.array(train_labels)
    # train_name_labels = np.array(train_name_labels)

    x_test = np.array(val_imgs[n:n + n_test])
    y_test = np.array(val_labels[n:n + n_test])
    # test_name_labels = np.array(val_name_labels[n:n + n_test])

    # train_name_labels, test_name_labels, val_name_labels
    return x_train, x_val, x_test, y_train, y_val, y_test, n_test, n


def divide_data(imgs, labels, name_labels, n_test=10):
    # to unorder samples
    random_seed = 35
    random.Random(random_seed).shuffle(labels)
    random.Random(random_seed).shuffle(name_labels)
    random.Random(random_seed).shuffle(imgs)
    n = len(imgs) - (1+n_test)

    x_train = np.array(imgs[: n])
    y_train = np.array(labels[: n])
    train_name_labels = np.array(name_labels[: n])

    x_test = np.array(imgs[n:n + n_test])
    y_test = np.array(labels[n:n + n_test])
    test_name_labels = np.array(name_labels[n:n + n_test])


    return n_test, n, x_test, x_train, y_test, y_train, train_name_labels, test_name_labels
