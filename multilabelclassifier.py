
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

# classes = []
# # _ = space ! = noise, ^ = partial, - = edge
# non_letters = ["_", "!", "^", "$", "&", "(", ")", "-"]
# for cL in list(string.ascii_lowercase):
#     classes.append(cL)
# for cU in list(string.ascii_uppercase):
#     classes.append(cU)
# for number in range(0, 10):
#     classes.append(number)
# for other in non_letters:
#     classes.append(other)

label_dict = {"word2idx": {}, "idx2word": []}
def prepare_data(imgs_dir):
    os.chdir(imgs_dir)
    y = []
    idx = 0

    imgs = []
    clean_titles = []
    for file in glob.glob("*.jpg"):
        img = scipy.misc.imread(file).astype(np.float32)

        if img.shape[0] == SIZE[0] and img.shape[1] == SIZE[1] and img.shape[2] == 3:
            imgs.append(img)
            clean_titles.append(re.sub(r"\([\d+]*\)", "", str(file.replace(".jpg", "").replace(" ", ""))))
        else:
            print("img size mismatch: {}".format(img.shape))


    # Add all file labels to dict, with indexes
    for title in clean_titles:
        for l in list(title): #.split('|'):
            if l in label_dict["idx2word"]:
                pass
            else:
                label_dict["idx2word"].append(l)
                label_dict["word2idx"][l] = idx
                idx += 1

    print(label_dict)

    # add multi-hot labels to overall labels?
    for title in clean_titles:
        letters = list(title)
        n_classes = len(label_dict["idx2word"])
        l = np.sum([np.eye(n_classes, dtype="uint8")[label_dict["word2idx"][s]]
                                                            for s in letters], axis=0)
        # print("letters: {}\nlabel: {}".format(letters, l))
        y.append(l)

    return imgs, y

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
print("1.1")
model.add(Conv2D(32, kernel_size=(5, 5), activation='relu',
                 input_shape=(SIZE[0], SIZE[1], 3)))
print("1.2")
model.add(Conv2D(32, (5, 5), activation='relu'))
print("1.3")
model.add(MaxPooling2D(pool_size=(4, 4)))
print("1.4")
model.add(Dropout(0.25))
print("1.5")
model.add(Conv2D(64, kernel_size=(5, 5), activation='relu'))
print("1.6")
model.add(Conv2D(64, (5, 5), activation='relu'))
print("1.7")
model.add(MaxPooling2D(pool_size=(4, 4)))
print("1.8")
model.add(Dropout(0.25))
print("1.9")
model.add(Flatten())
print("1.10")
model.add(Dense(128, activation='relu'))
print("1.11")
model.add(Dropout(0.5))
print("1.12")
model.add(Dense(58, activation='sigmoid'))


model.compile(loss='binary_crossentropy',
              optimizer=keras.optimizers.Adam(),
              metrics=['accuracy'])

# current 547 train and vaildate on 500, test examples for predictions on 47
# total imgs?
n_test = 5
n = len(dataset) -(1+n_test)

print("Beginning fit...")
model.fit(np.array(dataset[: n]), np.array(y[: n]), batch_size=2, epochs=3,
          verbose=1, validation_split=0.45)
print("spliting dataset")
X_test = dataset[n:n + n_test]
y_test = y[n:n + n_test]

print(len(X_test))
print("Ytest: ",len(y_test))

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
