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


def read_my_csv(file_name, delimiter='/'):
    """
    This function is used to pull specific label atrributes from a file,
    in addition to processing the input images.
    """
    print("reading data from csv: {} with delimiter '{}'".format(file_name, delimiter))
    path = os.getcwd() + "\\{}".format(file_name)
    df = pd.read_csv(path, delimiter=delimiter)
    labels = []
    eyes = np.eye(8, dtype="uint8")

    for label in df.Y:
        labels.append(eyes[label-1])

    imgs, name_labels, n_classes, name_label_dict, SIZE = prepare_data(os.getcwd()+"\\dataset")

    return imgs, labels, name_labels, n_classes, name_label_dict, SIZE

    # return imgs, labels, n_classes, label_dict, SIZE
