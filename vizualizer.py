def vizualize_layer(model, filter_index):

    # get the symbolic outputs of each "key" layer (we gave them unique names).
    layer_dict = dict([(layer.name, layer) for layer in model.layers])


    from keras import backend as K

    layer_name = 'conv2d_1'
     # can be any integer from 0 to 511, as there are 512 filters in that layer
    print("Layers: {}".format(layer_dict))
    # build a loss function that maximizes the activation
    # of the nth filter of the layer considered
    layer_output = layer_dict[layer_name].output
    loss = K.mean(layer_output[:, :, :, filter_index])

    # compute the gradient of the input picture wrt this loss
    grads = K.gradients(loss, input_img)[0]

    # normalization trick: we normalize the gradient
    grads /= (K.sqrt(K.mean(K.square(grads))) + 1e-5)

    # this function returns the loss and grads given the input picture
    iterate = K.function([input_img], [loss, grads])

    import numpy as np

    # we start from a gray image with some noise
    input_img_data = np.random.random((1, 3, 70, 60)) * 20 + 128.
    # run gradient ascent for 20 steps
    for i in range(20):
        loss_value, grads_value = iterate([input_img_data])
        input_img_data += grads_value * step

    from scipy.misc import imshow

    # util function to convert a tensor into a valid image
    def deprocess_image(x):
        # normalize tensor: center on 0., ensure std is 0.1
        x -= x.mean()
        x /= (x.std() + 1e-5)
        x *= 0.1

        # clip to [0, 1]
        x += 0.5
        x = np.clip(x, 0, 1)

        # convert to RGB array
        x *= 255
        x = x.transpose((1, 2, 0))
        x = np.clip(x, 0, 255).astype('uint8')
        return x

    img = input_img_data[0]
    img = deprocess_image(img)
    imshow('%s_filter_%d.png' % (layer_name, filter_index), img)
