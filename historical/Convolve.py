"""
This is the tricky part...
We need to pass every few pixels of the document to be convolved for every letter
This is going to be intense (processing)
To reduce the intensity of that processing, we Could
 1. select individual words to convolve, rather than the whole image  (risk of missing some)
 2. convolve entire image (slow, and risk of over classification)
 3. Select lines of words to convolve and reduce unnecessary excess <---I'm liking this idea, mixed with #5
 4. Figure out how to split words with templating so individual letters can be classified
 5. Pass every possible square through for classification.
    When the number of times a letter is classified correctly passes a threshold, label that region.

Current plan:
Select by rows to process (get width based on median width, and use horizontal dilation for better edge detection)
Rotate convolving frame to match angle of rows
Run through rows 2 pixels at a time and classify each space. Most conclusive spaces will maintain label


Convolutional Neural Network Estimator for Cursive, built with tf.layers."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

tf.logging.set_verbosity(tf.logging.INFO)

CLASS_N = 7
def cnn_model_fn(features, labels, mode):
  """Model function for CNN."""
  # Input Layer
  # Reshape X to 4-D tensor: [batch_size, width, height, channels]
  # MNIST images are 28x28 pixels, and have one color channel

  # TODO set shape based on average contour height
  input_layer = tf.reshape(features["x"], [-1, 26, 80, 1])
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


def main(unused_argv):
  # Load training and eval data
  mnist = tf.contrib.learn.datasets.load_dataset("mnist")
  print("print mnist: ", mnist)
  train_data = mnist.train.images  # Returns np.array
  train_labels = np.asarray(mnist.train.labels, dtype=np.int32)
  eval_data = mnist.test.images  # Returns np.array
  eval_labels = np.asarray(mnist.test.labels, dtype=np.int32)
  # images = input_data.read_data_sets("C:\\Users\\grant\\IS\\IS693R\\image_project\\images\\CensusRecords\\1900s\\isolated", one_hot=True)
  # print("print images: ", images)
  # train_data = ""
  # train_labels = ""
  # eval_data = ""
  # eval_labels = ""

  # Create the Estimator
  mnist_classifier = tf.estimator.Estimator(
      model_fn=cnn_model_fn, model_dir="/tmp/mnist_convnet_model/adjust1")

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
  mnist_classifier.train(
      input_fn=train_input_fn,
      steps=100, #20000
      hooks=[logging_hook])

  # Evaluate the model and print results
  eval_input_fn = tf.estimator.inputs.numpy_input_fn(
      x={"x": eval_data},
      y=eval_labels,
      num_epochs=1,
      shuffle=False)
  eval_results = mnist_classifier.evaluate(input_fn=eval_input_fn)
  print(eval_results)


if __name__ == "__main__":
  tf.app.run()
