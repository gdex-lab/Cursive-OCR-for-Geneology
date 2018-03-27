# Cursive-OCR-for-Geneology
Research findings and trained models associated with cursive handwriting recognition of antiquated records. Created to store and propagate findings from research at BYU during capstone semester, 2018. Emphasis given to genealogical records which need indexing.


If you are interested in contributing to this project, please reach out with a personalized invitation to https://www.linkedin.com/in/d-grant-dexter/.

# Overview
The attached dataset includes (1) full images, (2) sliced windows (from window_slider.py function), and (3) individual characters. You will also find custom modules for reading, vizualizing, and modeling with these image datasets. 

# Tips
For binary classification, use the sigmoid activiation on the final output layer, and a simple loss function like mean_squared_error (unless using one-hot encoding [0, 1] then softmax will work with categorical_crossentropy loss function). For multi-class models, use a softmax activation on the final output layer with a categorical_crossentropy loss. For multi-label classification, use a sigmoid function (because of independent probabilities) with a binary_crossentropy loss.

# Tools
The Keras preprocessor can be used to augment datasets. Take advantage of vizualizer.py when designing a network to vizualize the filters and pooling layers of your CNN. load_images_dataset.py has functions for reading in images from a folder with image names as labels, or reading a csv with image paths and labels on each row. Also use the divide_dataset function for shuffling your data and spliting a training, validation, and test dataset.
