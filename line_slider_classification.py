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
# import load_images_dataset


def prepare_data(imgs_dir, SIZE=(60,25), skips=[".jpg", " "]):

    os.chdir(imgs_dir)
    label_dict = {"word2idx": {}, "idx2word": []}
    labels = []
    idx = 0
    imgs = []
    clean_titles = []
    label_cardinality = {}

    for file in glob.glob("*/*.jpg", recursive=True):
        # img = scipy.misc.imread(file).astype(np.unit8)
        img = cv2.imread(file)
        # img = cv2.erode(img,kernel,iterations = 2)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        if img.shape[0] == SIZE[0] and img.shape[1] == SIZE[1]: # and img.shape[2] == SIZE[2]:
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
        for l in list(title):
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


x, y, n_classes, label_dict, size = prepare_data('C:\\Users\\grant\\Repos\\Cursive-OCR-for-Geneology\\dataset\\ready_singles')


def divide_data(imgs, labels, name_labels, n_test=10):
    # to unorder samples
    random_seed = 4
    random.Random(random_seed).shuffle(labels)
    # random.Random(random_seed).shuffle(name_labels)
    random.Random(random_seed).shuffle(imgs)
    n = len(imgs) - (1+n_test)

    x_train = np.array(imgs[: n])
    y_train = np.array(labels[: n])
    # train_name_labels = np.array(name_labels[: n])

    x_test = np.array(imgs[n:n + n_test])
    y_test = np.array(labels[n:n + n_test])
    # test_name_labels = np.array(name_labels[n:n + n_test])

    # first_img =  np.squeeze(x_train[5])
    # print("Graph label: ", y_train[5])
    # plt.imshow(first_img)
    # plt.show()
    return n_test, n, x_test, x_train, y_test, y_train


n_test, n_classes, x_test, x_train, y_test, y_train = divide_data(x, y, label_dict, n_test=10)


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
