import custom_models
import load_images_dataset
from keras.models import model_from_yaml
# from keras.utils import plot_model


p_data = load_images_dataset.PreparedData()
p_data.set_size((60, 25))
p_data.process()

epochs = 35
batch_size = 64
# print(p_data.dataset['y_val'][:7])
test_loss = 13
test_accuracy = 0
current_epoch = 0
# model = custom_models.cursive_cnn(p_data.dataset['x_train'], p_data.dataset['y_train'],
# p_data.dataset['x_val'], p_data.dataset['y_val'],
# p_data.size, p_data.n_classes,
# epochs=epochs, batch_size=batch_size)

print(p_data.dataset['y_test'])

while test_loss > .7:
    print("Current epochs: {}".format(current_epoch))
    if (current_epoch > 50 and test_accuracy < 0.25 and test_loss > 10) or current_epoch == 0:

        print("loading previous model")

        yaml_file = open('C:\\Users\\grant\\Repos\\Cursive-OCR-for-Geneology\\slider_cnn_current.yaml', 'r')
        loaded_model_yaml = yaml_file.read()
        yaml_file.close()
        model = model_from_yaml(loaded_model_yaml)
        print("Loaded.")
        # load weights into new model
        model.load_weights("C:\\Users\\grant\\Repos\\Cursive-OCR-for-Geneology\\model2_current.h5")
        print("Loaded weights from disk")
        model.compile(loss='categorical_crossentropy',
                      optimizer='Adadelta',
                      metrics=['accuracy'])
        print("Compiled\nFitting...")
        model.fit(p_data.dataset['x_train'], p_data.dataset['y_train'],
                  batch_size=batch_size,
                  epochs=epochs,
                  verbose=1,
                  # validation_split=0.4,
                  validation_data=(p_data.dataset['x_val'], p_data.dataset['y_val'])
                  )
        print("Fit")
        # print("(re)Initializing model")
        # # if sufficient attemps, and poor accuracy, re-initialize, or if first time
        # model = custom_models.bw_cnn(p_data.dataset['x_train'], p_data.dataset['y_train'],
        # # p_data.dataset['x_val'], p_data.dataset['y_val'],
        # p_data.size, p_data.n_classes,
        # epochs=epochs, batch_size=batch_size)
        # current_epoch = 0

    else:
        print("Saving and loading progress")
        # if the model is performing well, or less than 100 epochs, save the weights, and keep training
        # serialize model to YAML
        model_yaml = model.to_yaml()
        with open("C:\\Users\\grant\\Repos\\Cursive-OCR-for-Geneology\\slider_cnn_3.yaml", "w") as yaml_file:
            yaml_file.write(model_yaml)
        # serialize weights to HDF5
        model.save_weights("C:\\Users\\grant\\Repos\\Cursive-OCR-for-Geneology\\model2_{}_{}.h5".format(test_loss, test_accuracy))
        print("Saved model to disk")

        # load YAML and create model
        yaml_file = open('C:\\Users\\grant\\Repos\\Cursive-OCR-for-Geneology\\slider_cnn_3.yaml', 'r')
        loaded_model_yaml = yaml_file.read()
        yaml_file.close()
        model = model_from_yaml(loaded_model_yaml)
        # load weights into new model
        model.load_weights("C:\\Users\\grant\\Repos\\Cursive-OCR-for-Geneology\\model2_{}_{}.h5".format(test_loss, test_accuracy))
        print("Loaded model from disk")
        model.compile(loss='categorical_crossentropy',
                      optimizer='Adadelta',
                      metrics=['accuracy'])
        model.fit(p_data.dataset['x_train'], p_data.dataset['y_train'],
                  batch_size=batch_size,
                  epochs=epochs,
                  verbose=1,
                  # validation_split=0.4,
                  validation_data=(p_data.dataset['x_val'], p_data.dataset['y_val'])
                  )


    score = model.evaluate(p_data.dataset['x_test'], p_data.dataset['y_test'], verbose=1)
    test_loss = score[0]
    test_accuracy = score[1]
    print('Test accuracy:', test_accuracy)
    print('Test loss:', test_loss)
    current_epoch += epochs



model_yaml = model.to_yaml()
with open("C:\\Users\\grant\\Repos\\Cursive-OCR-for-Geneology\\slider_cnn3{}_{}.yaml".format(test_loss, test_accuracy), "w") as yaml_file:
    yaml_file.write(model_yaml)
# serialize weights to HDF5
model.save_weights("C:\\Users\\grant\\Repos\\Cursive-OCR-for-Geneology\\model3{}_{}.h5".format(test_loss, test_accuracy))
print("Saved model to disk")
