
# from vizualizer import vizualize_layer
import scipy
import matplotlib.pyplot as plt
import cv2
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, Conv1D, MaxPooling2D,MaxPooling1D, AveragePooling2D, GlobalMaxPooling2D
import keras
import numpy as np
# vizualize_layer(model, scipy.misc.imread('C:\\Users\\grant\\Repos\\Cursive-OCR-for-Geneology\\vizualize_examplery_images\\_lan.jpg').astype(np.float32))
img1 = cv2.imread('C:\\Users\\grant\\Repos\\Cursive-OCR-for-Geneology\\vizualize_examplery_images\\_lan.jpg')
# kernel = np.ones((5,9),np.uint8)
# img1 = cv2.erode(img1,kernel,iterations = 1) # erosion is actually dilation in this case
# img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)

# print(img.shape)


model = Sequential()
# model.add(MaxPooling1D(pool_size=(2), input_shape=img1.shape))
model.add(Conv2D(4,
                 kernel_size=(3,3),
                 activation='sigmoid',
                 input_shape=img1.shape))
# model.add(Conv2D(32,
#                  kernel_size=(2),
#                  activation='sigmoid',
#                  input_shape=img1.shape))
model.add(MaxPooling2D(15, 17))

img_batch = np.expand_dims(img1, axis=0)
conv_img = model.predict(img_batch)
print("here 2")

print("here2")
def vizualize_layer(img_batch):
    print(img_batch)
    img2 = np.squeeze(img_batch, axis=0)
    print("\n\n")
    # print(img)
    print("here 3")
    # print(img.shape)
    fig=plt.figure(figsize=(4, 4))
    fig.add_subplot(1, 2, 1)
    plt.imshow(img1)
    fig.add_subplot(1, 2, 2)
    plt.imshow(img2)
    plt.show()

vizualize_layer(conv_img)







# import numpy as np
# from keras import backend as K
# from scipy.misc import imshow
#
#
# def vizualize_layer(model, input_img, layer_name='conv2d_1', filter_index=1):
#
#     # get the symbolic outputs of each "key" layer (we gave them unique names).
#     layer_dict = dict([(layer.name, layer) for layer in model.layers])
#
#      # can be any integer from 0 to 511, as there are 512 filters in that layer
#     print("Layers: {}".format(layer_dict))
#     # build a loss function that maximizes the activation
#     # of the nth filter of the layer considered
#     layer_output = layer_dict[layer_name].output
#
#     print(layer_output)
#     print(layer_output[:, :, :, filter_index])
#
#     loss  = K.mean(layer_output[:, :, :, filter_index])
#     print("Loss = {}:".format(loss))
#     # compute the gradient of the input picture wrt this loss
#     grads = K.gradients(layer_output, input_img)[0]
#
#     # print(input_img)
#     print("Grads: {}".format(grads))
#
#     # normalization trick: we normalize the gradient
#     grads /= (K.sqrt(K.mean(K.square(grads))) + 1e-5)
#     print("here")
#
#     # this function returns the loss and grads given the input picture
#     iterate = K.function([input_img], [loss, grads])
#     print("here2")
#
#     # we start from a gray image with some noise
#     input_img_data = np.random.random((1, 3, 70, 60)) * 20 + 128.
#     # run gradient ascent for 20 steps
#     for i in range(20):
#         loss_value, grads_value = iterate([input_img_data])
#         input_img_data += grads_value * step
#
#
#     # util function to convert a tensor into a valid image
#     def deprocess_image(x):
#         # normalize tensor: center on 0., ensure std is 0.1
#         x -= x.mean()
#         x /= (x.std() + 1e-5)
#         x *= 0.1
#
#         # clip to [0, 1]
#         x += 0.5
#         x = np.clip(x, 0, 1)
#
#         # convert to RGB array
#         x *= 255
#         x = x.transpose((1, 2, 0))
#         x = np.clip(x, 0, 255).astype('uint8')
#         return x
#
#     img = input_img_data[0]
#     img = deprocess_image(img)
#     imshow('%s_filter_%d.png' % (layer_name, filter_index), img)
