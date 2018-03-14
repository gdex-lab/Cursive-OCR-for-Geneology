import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D

def basic_cnn(activation_1, loss, x_train, y_train, \
                    input_shape, n_classes, base_layers=3,
                    epochs=3, learning_rate=64, \
                    conv_size=3, pool_size=2):

    model = Sequential()
    model.add(Conv2D(base_layers,
                     kernel_size=(conv_size, conv_size),
                     activation=activation_1,
                     input_shape=input_shape))


    #tanh offering more specific vals, rather than 1 0
    # model.add(Conv2D(4, (conv_size, conv_size), activation=activation_1)) # relu
    # model.add(MaxPooling2D(pool_size=(pool_size, pool_size)))
    # model.add(Dropout(0.25))

    # #bonus
    # model.add(Conv2D(base_layers, (3, 3), activation=activation_1)) # relu
    # model.add(MaxPooling2D(pool_size=(2, 2)))
    # model.add(Dropout(0.25))

    model.add(Flatten())
    # # model.add(MaxPooling2D(pool_size=(pool_size, pool_size)))
    # model.add(Dense(8, activation=activation_1))
    # model.add(Dropout(0.5))
    model.add(Dense(2, activation='sigmoid'))

     # 'categorical_crossentropy' <- supposedly for multi-class, not multi label: https://stats.stackexchange.com/questions/260505/machine-learning-should-i-use-a-categorical-cross-entropy-or-binary-cross-entro
    model.compile(loss=loss,
                    optimizer=keras.optimizers.Adam(), #.Adam(), Adadelta()
                    metrics=['categorical_accuracy', 'mae']
                    )
                    #'binary_crossentropy' : supposedely ideal for multi label, current .5 test accuracy, but no letters predicted
                    # 'mean_squared_error' : all same, 1s

    model.fit(x_train, y_train,
              batch_size=64, #128
              epochs=epochs,
              verbose=1,
              # validation_data=(x_test, y_test)
              validation_split=0.4
              )
    return model


def seven_layer_cnn(activation_1, activation_2, loss, x_train, y_train, \
                    input_shape, n_classes, base_layers, epochs=3):

    model = Sequential()
    model.add(Conv2D(int(n_classes/2),
                     kernel_size=(3, 3),
                     activation=activation_1,
                     input_shape=input_shape))
    #tanh offering more specific vals, rather than 1 0
    model.add(Conv2D(n_classes, (3, 3), activation=activation_1)) # relu
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    #bonus
    model.add(Conv2D(n_classes, (3, 3), activation=activation_1)) # relu
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(n_classes*2, activation=activation_1))
    model.add(Dropout(0.5))
    model.add(Dense(n_classes, activation=activation_2))

     # 'categorical_crossentropy' <- supposedly for multi-class, not multi label: https://stats.stackexchange.com/questions/260505/machine-learning-should-i-use-a-categorical-cross-entropy-or-binary-cross-entro
    model.compile(loss=loss,
    #'binary_crossentropy' : supposedely ideal for multi label, current .5 test accuracy, but no letters predicted
    # 'mean_squared_error' : all same, 1s
                  optimizer=keras.optimizers.Adam(), #.Adam(), Adadelta()
                  metrics=['categorical_accuracy', 'accuracy', 'mae'])

    model.fit(x_train, y_train,
              batch_size=learning_rate, #128
              epochs=epochs,
              verbose=1,
              # validation_data=(x_test, y_test)
              validation_split=0.4
              )
    return model
