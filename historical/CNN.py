"""Convolutional Neural Network Estimator for Cursive, built with tf.layers."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import glob, os
import cv2
import numpy as np
import tensorflow as tf

from tensorflow.contrib.learn.python.learn.datasets import base
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import random_seed

tf.logging.set_verbosity(tf.logging.INFO)

if os.name == 'nt':
    env = 0
else:
    env = 1

CLASS_N = 9
SZ_W = 40
SZ_H = 60

def cnn_model_fn(features, labels, mode):
    """Model function for CNN."""
    # Input Layer
    # Reshape X to 4-D tensor: [batch_size, width, height, channels]
    # MNIST images are 28x28 pixels, and have one color channel
    input_layer = tf.reshape(features["x"], [-1, SZ_W, SZ_H, 1])
    print(input_layer)
    # Convolutional Layer #1
    # Computes 32 features using a 5x5 filter with ReLU activation.
    # Padding is added to preserve width and height.
    # Input Tensor Shape: [batch_size, 26, 80, 1]
    # Output Tensor Shape: [batch_size, 26, 80, 32]
    conv1 = tf.layers.conv2d(
      inputs=input_layer,
      filters=32,
      kernel_size=[5, 5],
      padding="same",
      activation=tf.nn.relu)
    print(conv1)
    # Pooling Layer #1
    # First max pooling layer with a 2x2 filter and stride of 2
    # Input Tensor Shape: [batch_size, 26, 80, 32]
    # Output Tensor Shape: [batch_size, 13, 40, 32]
    pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)
    print(pool1)
    # Convolutional Layer #2
    # Computes 64 features using a 5x5 filter.
    # Padding is added to preserve width and height.
    # Input Tensor Shape: [batch_size, 13, 40, 32]
    # Output Tensor Shape: [batch_size, 13, 40, 64]
    conv2 = tf.layers.conv2d(
      inputs=pool1,
      filters=64,
      kernel_size=[5, 5],
      padding="same",
      activation=tf.nn.relu)
    print(conv2)
    # Pooling Layer #2
    # Second max pooling layer with a 2x2 filter and stride of 2
    # Input Tensor Shape: [batch_size, 13, 40, 64]
    # Output Tensor Shape: [batch_size, 7, 20, 64]
    # was pool size 2, 2
    pool2 = conv2 #tf.layers.max_pooling2d(inputs=conv2, pool_size=[1, 2], strides=2)
    print(pool2)
    # Flatten tensor into a batch of vectors
    # Input Tensor Shape: [batch_size, 7, 20, 64?]
    # Output Tensor Shape: [batch_size, 7 * 20 * 64]
    flattened_size = (int(pool2.shape[1]) * int(pool2.shape[2]) * int(pool2.shape[3]))
    pool2_flat = tf.reshape(pool2, [-1, flattened_size])
    print(pool2_flat)
    # Dense Layer
    # Densely connected layer with 1024 neurons
    # Input Tensor Shape: [batch_size, 7 * 20 * 64]
    # Output Tensor Shape: [batch_size, 8960]
    dense = tf.layers.dense(inputs=pool2_flat, units=flattened_size, activation=tf.nn.relu)
    print(dense)
    # Add dropout operation; 0.6 probability that element will be kept
    dropout = tf.layers.dropout(
      inputs=dense, rate=0.4, training=mode == tf.estimator.ModeKeys.TRAIN)
    print(dropout)
    # Logits layer
    # Input Tensor Shape: [batch_size, 16640]
    # Output Tensor Shape: [batch_size, 10]
    logits = tf.layers.dense(inputs=dropout, units=CLASS_N)
    print("logits: ", logits)


    predictions = {
      # Generate predictions (for PREDICT and EVAL mode)
      "classes": tf.argmax(input=logits, axis=1),
      # Add `softmax_tensor` to the graph. It is used for PREDICT and by the
      # `logging_hook`.
      "probabilities": tf.nn.softmax(logits, name="softmax_tensor")
    }
    print(predictions)

    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)
    print(1.0)
    # Calculate Loss (for both TRAIN and EVAL modes)
    onehot_labels = tf.one_hot(indices=tf.cast(labels, tf.int32), depth=CLASS_N)
    print("1.0.1")
    print(onehot_labels)
    loss = tf.losses.softmax_cross_entropy(onehot_labels=onehot_labels, logits=logits)
    print(1.1)
    # Configure the Training Op (for TRAIN mode)
    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
        train_op = optimizer.minimize(
            loss=loss,
            global_step=tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)
    print(1.2)
    # Add evaluation metrics (for EVAL mode)
    eval_metric_ops = {
      "accuracy": tf.metrics.accuracy(
          labels=labels, predictions=predictions["classes"])}
    return tf.estimator.EstimatorSpec(
                    mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)

def read_data_sets(train_dir,
                   one_hot=False,
                   dtype=dtypes.float32,
                   reshape=True,
                   validation_size=5000,
                   seed=None):


    # Pass in the images here.
    # TRAIN_IMAGES = 'train-images-idx3-ubyte.gz'
    # TRAIN_LABELS = 'train-labels-idx1-ubyte.gz'
    # TEST_IMAGES = 't10k-images-idx3-ubyte.gz'
    # TEST_LABELS = 't10k-labels-idx1-ubyte.gz'
    #
    # local_file = base.maybe_download(TRAIN_IMAGES, train_dir,
    #                                  SOURCE_URL + TRAIN_IMAGES)
    # with open(local_file, 'rb') as f:
    #   train_images = extract_images(f)
    #
    # local_file = base.maybe_download(TRAIN_LABELS, train_dir,
                                   # SOURCE_URL + TRAIN_LABELS)
    # with open(local_file, 'rb') as f:
    #   train_labels = extract_labels(f, one_hot=one_hot)
    # print('mnist train labels: ', train_labels.shape)
    # print('example 1: ', train_labels[1])
    #
    # local_file = base.maybe_download(TEST_IMAGES, train_dir,
    #                                  SOURCE_URL + TEST_IMAGES)
    # with open(local_file, 'rb') as f:
    #   test_images = extract_images(f)
    #
    # local_file = base.maybe_download(TEST_LABELS, train_dir,
    #                                  SOURCE_URL + TEST_LABELS)
    # with open(local_file, 'rb') as f:
    #   test_labels = extract_labels(f, one_hot=one_hot)
    #
    # # if not 0 <= validation_size <= len(train_images):
    # #   raise ValueError(
    # #       'Validation size should be between 0 and {}. Received: {}.'
    # #       .format(len(train_images), validation_size))
    #
    # validation_images = train_images[:validation_size]
    # validation_labels = train_labels[:validation_size]
    # train_images = train_images[validation_size:]
    # train_labels = train_labels[validation_size:]

    print("loading windows...")
    # This is where the auto_transcribe functions will input sliced windows

    if env == 0:
        os.chdir("C:\\Users\\grant\\Repos\\Cursive-OCR-for-Geneology\\dataset")
    else:
        os.chdir("/home/ubuntu/Cursive-OCR-for-Geneology/dataset")
    windows = []
    labels = []
    label_names = ["a", "e", "i",  "o", "u", "h", "n", "t", "other"]
    total_imgs = 0
    for file in glob.glob("*.jpg"):
        # if label_names in str(file):
        img = cv2.imread(file)

        # Convert img to grayscale to prep for black and white
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) #grey increased accuracy by .3%
        # Convert to black and white based on automatice OTSU threshold
        # (thresh, img) = cv2.threshold(img, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU) # b&w increased it by 20%

        windows.append(img)

        str_label = str(file)

        # instanciate int_label
        # int_label =  [0, 0, 0, 0, 0, 0, 0]
        # if str_label == 'll':
        #   int_label =  [1, 0, 0, 0, 0, 0, 0]
        # elif str_label == 'ly':
        #   int_label =  [0, 1, 0, 0, 0, 0, 0]
        # elif str_label == 'lh':
        #   int_label =  [0, 0, 1, 0, 0, 0, 0]
        # elif str_label == 'lo':
        #   int_label =  [0, 0, 0, 1, 0, 0, 0]
        # elif str_label == 'le':
        #   int_label =  [0, 0, 0, 0, 1, 0, 0]
        # elif str_label == 'la':
        #   int_label =  [0, 0, 0, 0, 0, 1, 0]
        # elif str_label == 'ln':
        #   int_label =  [0, 0, 0, 0, 0, 0, 1]
        # labels.append(int_label)
        int_label =  0
        if "a" in str_label:
            int_label =  0
        elif "e" in str_label:
            int_label =  1
        elif "i" in str_label:
            int_label =  2
        elif "o" in str_label:
            int_label =  3
        elif "u" in str_label:
            int_label =  4
        elif "h" in str_label:
            int_label =  5
        elif "n" in str_label:
            int_label =  6
        elif "t" in str_label:
            int_label =  7
        elif "noise" in str_label:
            int_label =  8
        labels.append(int_label)
        total_imgs += 1


    windows = np.array(windows)
    labels = np.array(labels)
    print('my labels: ', labels.shape)
    windows = windows.reshape(-1, SZ_H, SZ_W) # was originally reversed. Correct is H then W

    # shuffle digits
    rand = np.random.RandomState(5)
    shuffle = rand.permutation(181) # len of windows turns out larger than 84?? 252??
    # print(rand, '\n\n\n',len(windows),'\n\n\n', shuffle)
    windows, labels = windows[shuffle], labels[shuffle]
    # print(windows, labels)

    # total sie = 181 with 7 classes currently
    # total_imgs = 547
    val_size = int(.2 * total_imgs)

    validation_images = windows[:val_size]
    validation_labels = labels[:val_size]
    train_images = windows[val_size:]
    train_labels = labels[val_size:]

    # if I validate with training data, I am still only getting 26% accuracy
    # try with black and white images next to see what changes.

    # come back and address the 20 test/training later? is okay that they are validation?
    test_images = train_images #validation_images # windows[:90]
    test_labels = train_labels #validation_labels # labels[:90]

    options = dict(dtype=dtype, reshape=reshape, seed=seed)

    train = DataSet(train_images, train_labels, **options)
    validation = DataSet(validation_images, validation_labels, **options)
    test = DataSet(test_images, test_labels, **options)

    return base.Datasets(train=train, validation=validation, test=test), total_imgs


def images_to_tensors(train_dir="C:\\Users\\grant\\Repos\\Cursive-OCR-for_Geneology"):
  if env == 1:
    train_dir="/home/ubuntu/Cursive-OCR-for-Geneology/"
  return read_data_sets(train_dir)

class DataSet(object):
    """Construct a DataSet.
    one_hot arg is used only if fake_data is true.  `dtype` can be either
    `uint8` to leave the input as `[0, 255]`, or `float32` to rescale into
    `[0, 1]`.  Seed arg provides for convenient deterministic testing.
    """
    def __init__(self,
               images,
               labels,
               fake_data=False,
               one_hot=False,
               dtype=dtypes.float32,
               reshape=True,
               seed=None):
        # If op level seed is not set, use whatever graph level seed is returned
        seed1, seed2 = random_seed.get_seed(seed)
        np.random.seed(seed1 if seed is None else seed2)
        dtype = dtypes.as_dtype(dtype).base_dtype
        if dtype not in (dtypes.uint8, dtypes.float32):
            raise TypeError('Invalid image dtype %r, expected uint8 or float32' %
                          dtype)
        if fake_data:
            self._num_examples = 10000
            self.one_hot = one_hot
        else:
            assert images.shape[0] == labels.shape[0], (
              'images.shape: %s labels.shape: %s' % (images.shape, labels.shape))
            self._num_examples = images.shape[0]

              # Convert shape from [num examples, rows, columns, depth]
              # to [num examples, rows*columns] (assuming depth == 1)

        if reshape:
            print('shape: ', images.shape)
            # assert images.shape[3] == 1
            images = images.reshape(images.shape[0],
                                    images.shape[1] * images.shape[2])
            print('reshaped: ', images.shape)

        if dtype == dtypes.float32:
            # Convert from [0, 255] -> [0.0, 1.0].
            images = images.astype(np.float32)
            images = np.multiply(images, 1.0 / 255.0)
        self._images = images
        self._labels = labels
        self._epochs_completed = 0
        self._index_in_epoch = 0

    @property
    def images(self):
        return self._images

    @property
    def labels(self):
        return self._labels

    @property
    def num_examples(self):
        return self._num_examples

    @property
    def epochs_completed(self):
        return self._epochs_completed

    def next_batch(self, batch_size, fake_data=False, shuffle=True):
        """Return the next `batch_size` examples from this data set."""
        if fake_data:
            fake_image = [1] * 784
            # if self.one_hot:
            #   fake_label = [1] + [0] * 9
            # else:
            #   fake_label = 0
            # return [fake_image for _ in xrange(batch_size)], [
            #     fake_label for _ in xrange(batch_size)
            # ]
        start = self._index_in_epoch
        # Shuffle for the first epoch
        if self._epochs_completed == 0 and start == 0 and shuffle:
            perm0 = np.arange(self._num_examples)
            np.random.shuffle(perm0)
            self._images = self.images[perm0]
            self._labels = self.labels[perm0]
        # Go to the next epoch
        if start + batch_size > self._num_examples:
            # Finished epoch
            self._epochs_completed += 1
            # Get the rest examples in this epoch
            rest_num_examples = self._num_examples - start
            images_rest_part = self._images[start:self._num_examples]
            labels_rest_part = self._labels[start:self._num_examples]
            # Shuffle the data
            if shuffle:
                perm = np.arange(self._num_examples)
                np.random.shuffle(perm)
                self._images = self.images[perm]
                self._labels = self.labels[perm]
            # Start next epoch
            start = 0
            self._index_in_epoch = batch_size - rest_num_examples
            end = self._index_in_epoch
            images_new_part = self._images[start:end]
            labels_new_part = self._labels[start:end]
            return np.concatenate((images_rest_part, images_new_part), axis=0) , np.concatenate((labels_rest_part, labels_new_part), axis=0)
        else:
            self._index_in_epoch += batch_size
            end = self._index_in_epoch
            return self._images[start:end], self._labels[start:end]


def main(unused_argv):
  # Load training and eval data
  dataset, total_imgs = images_to_tensors()
  print("print dataset: ", dataset)
  train_data = dataset.train.images  # Returns np.array
  train_labels = np.asarray(dataset.train.labels, dtype=np.int32)
  eval_data = dataset.test.images  # Returns np.array
  eval_labels = np.asarray(dataset.test.labels, dtype=np.int32)
  # images = input_data.read_data_sets("C:\\Users\\grant\\IS\\IS693R\\image_project\\images\\CensusRecords\\1900s\\isolated", one_hot=True)
  # print("print images: ", images)
  # train_data = ""
  # train_labels = ""
  # eval_data = ""
  # eval_labels = ""

  # Create the Estimator
  if env == 1:
    model_dir="/home/ubuntu/Cursive-OCR-for-Geneology/adjust_{}_rev1".format(total_imgs)
  else:
    model_dir="C:\\Users\\grant\\Repos\\Cursive-OCR-for-Geneology\\adjust_{}_rev1".format(total_imgs)

  cursive_classifier = tf.estimator.Estimator(
      model_fn=cnn_model_fn, model_dir=model_dir)

  # Set up logging for predictions
  # Log the values in the "Softmax" tensor with label "probabilities"
  tensors_to_log = {"probabilities": "softmax_tensor"}
  logging_hook = tf.train.LoggingTensorHook(
      tensors=tensors_to_log, every_n_iter=50)

  # Train the model
  train_input_fn = tf.estimator.inputs.numpy_input_fn(
      x={"x": train_data},
      y=train_labels,
      batch_size=50,
      num_epochs=None,
      shuffle=True)
  cursive_classifier.train(
      input_fn=train_input_fn,
      steps=100, #20000
      hooks=[logging_hook])

  # Evaluate the model and print results
  eval_input_fn = tf.estimator.inputs.numpy_input_fn(
      x={"x": eval_data},
      y=eval_labels,
      num_epochs=1,
      shuffle=False)
  eval_results = cursive_classifier.evaluate(input_fn=eval_input_fn)
  print(eval_results)


if __name__ == "__main__":
  tf.app.run()
