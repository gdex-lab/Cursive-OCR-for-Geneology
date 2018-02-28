
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

SIZE = (60, 70)


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


label_dict = {"word2idx": {}, "idx2word": []}
def prepare_data(imgs_dir):
    os.chdir(imgs_dir)
    y = []
    idx = 0
    skips = [".jpg", " ", "@", "+", "]", "[", ")", "(", "_",
    "$", "z", "j", "b", "k", "v", "w", # less than 50
    "A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O",
    "P", "Q", "R","S", "T", "U", "V", "W", "X", "Y", "Z",
            ".", ",", "0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]
    imgs = []
    clean_titles = []
    label_cardinality = {}
    for file in glob.glob("*/*.jpg", recursive=True):
        img = scipy.misc.imread(file).astype(np.float32)

        if img.shape[0] == SIZE[0] and img.shape[1] == SIZE[1] and img.shape[2] == 3:
            clean_title = str(file.split('\\')[1])
            clean_title = re.sub(r"\([\d+]*\)", "", clean_title)
            for lb in skips:
                clean_title = clean_title.replace(lb, "")

            if len(clean_title) > 0:
                imgs.append(img)
                # print(clean_title)
                clean_titles.append(clean_title)
        else:
            print("img size mismatch: {}".format(img.shape))


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
        y.append(l)

    # print(label_cardinality)
    for l in sorted(label_cardinality):
        print(l, ": ", label_cardinality[l])

    return imgs, y, n_classes

# dataset = imgs, label_dict = word2indx:, indx2word:, ids = img titles, y = list of sumed classes and label indexes
# dataset, y, label_dict, ids =  prepare_data(data, img_dict, size=SIZE)
dataset, y, n_classes =  prepare_data(path)


print("shuffling dataset")
print("Number of classes: {}".format(n_classes))
# to unorder samples
random_seed = 4
random.Random(random_seed).shuffle(y)
random.Random(random_seed).shuffle(dataset)


# def show_images(ids, cols = 2, titles = None):
#     fig = plt.figure()
#     n_images = len(ids)
#     n = 0
#     for id in ids:
#         plt.imshow(dataset[id])
#         a = fig.add_subplot(cols, np.ceil(n_images/float(cols)), n + 1)
#         # print(y[id])
#         # print(label_dict["idx2word"])
#         # print(np.where(y[id]>0))
#         actuals =""
#         for index in np.where(y[id]==1)[0]:
#             # print(index)
#             actuals += " {}".format(label_dict["idx2word"][index])
#         a.set_title(actuals)
#         n+=1
#         # for n, (image, title) in enumerate(zip(ids, titles)):
#         #     if image.ndim == 2:
#         #         plt.gray()
#         #     plt.imshow(image)
#     fig.set_size_inches(np.array(fig.get_size_inches()) * n_images)
#     mng = plt.get_current_fig_manager()
#     mng.window.state('zoomed')
#     plt.show()


def return_labels(id):
    labels = ""
    for index in np.where(y[id]>0)[0]:
        # print(index)
        labels += " {}".format(label_dict["idx2word"][index])
    # print("Returning labels: ", labels)
    return labels

def show_img(id):
    plt.suptitle(return_labels(id))
    plt.imshow(dataset[id])
    plt.ylabel(return_labels(id))
    plt.show()

# show_img(0)
# show_img(1)
# show_img(2)
# show_img(3)

# ------------------------------------



















print("model.fit DONE. Moving on to pred...")
pred = model.predict(np.array(X_test))
print("predictions finished")

for i in range (0, len(X_test)):
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
        if pred[i][i2] > 0.3:
            print("\"{}\":{}".format(label_dict["idx2word"][i2], pred[i][i2]))
    print("--------------------------------------")
