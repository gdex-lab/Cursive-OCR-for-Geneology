import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv1D, Conv2D, MaxPooling2D, MaxPooling1D, AveragePooling2D
from keras.optimizers import SGD

def basic_cnn(activation_1, loss, x_train, y_train, x_val, y_val,\
                    input_shape, n_classes, base_layers=3,
                    epochs=3, batch_size=64, \
                    conv_size=3, pool_size=2):
    print("input shape: {}".format(input_shape))

    model_aug = Sequential()
    # model_aug.add(Conv1D(7,
    #                  kernel_size=(1),
    #                  activation='sigmoid',
    #                  input_shape=input_shape))
    # model_aug.add(MaxPooling1D(4,4))
    # model_aug.add(Conv1D(32,
    #                  kernel_size=(2),
    #                  activation='sigmoid',
    #                  input_shape=input_shape))
    # model_aug.add(MaxPooling1D(12, 1))

    model_aug.add(Conv2D(4, (3, 3), activation='relu', padding='same', name='conv_1',
                     input_shape=(60, 70, 3)))
    model_aug.add(MaxPooling2D((15, 17), name='maxpool_1'))
    # model_aug.add(Conv2D(32, (3, 3), activation='relu', padding='same', name='conv_2'))
    # model_aug.add(MaxPooling2D((2, 2), name='maxpool_2'))
    model_aug.add(Flatten())
    # model_aug.add(Dropout(0.25))
    # model_aug.add(Dense(256, activation='relu', name='dense_1'))
    model_aug.add(Dense(8, activation='relu', name='dense_2'))
    model_aug.add(Dense(1, activation='sigmoid', name='output'))
    model_aug.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    # model.compile(loss='categorical_crossentropy', optimizer='Adam',  metrics=['categorical_accuracy'])


    model_aug.fit(x_train, y_train,
              batch_size=batch_size,
              epochs=epochs,
              verbose=1,
              validation_data=(x_val, y_val)
              )
    return model_aug


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
