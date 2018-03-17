import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv1D, Conv2D, MaxPooling2D, MaxPooling1D, AveragePooling2D
from keras.optimizers import SGD

def basic_cnn(activation_1, loss, x_train, y_train, \
                    input_shape, n_classes, base_layers=3,
                    epochs=3, batch_size=64, \
                    conv_size=3, pool_size=2):
    print("input shape: {}".format(input_shape))



    model = Sequential()
    model.add(Conv1D(3,
    kernel_size=(1),
    activation='sigmoid',
    input_shape=input_shape))
    # model.add(Conv1D(36,
    # kernel_size=(2),
    # activation='sigmoid',
    # input_shape=input_shape))

    model.add(Flatten())
    model.add(Dense(15, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(15, activation='relu'))

    # For a binary classifier you can either use a sigmoid activation with the
    # "binary_crossentropy" loss (and one output layer), or put two output units at the last layer,
    #  keep using softmax and change the loss to categorical_crossentropy.
    # 'categorical_crossentropy' <- supposedly for multi-class, not multi label: https://stats.stackexchange.com/questions/260505/machine-learning-should-i-use-a-categorical-cross-entropy-or-binary-cross-entro
    # 'binary_crossentropy' : supposedely ideal for multi label, current .5 test accuracy, but no letters predicted
    # 'mean_squared_error' : all same, 1s
    # model.add(Dense(24, activation='tanh'))
    # model.add(Dense(12, activation='tanh'))
    # model.add(Dropout(0.25))
    model.add(Dense(2, activation='softmax'))



    model.compile(loss='categorical_crossentropy', optimizer='Adam',  metrics=['categorical_accuracy'])




    #
    # model.compile(loss='categorical_crossentropy',
    #                 optimizer=keras.optimizers.Adam(), #.Adam(), Adadelta()
    #                 metrics=['accuracy', 'mae']
    #                 )

    model.fit(x_train, y_train,
              batch_size=batch_size,
              epochs=epochs,
              verbose=1,
              validation_split=0.3
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
