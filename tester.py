from keras.models import model_from_yaml
import load_images_dataset

p_data = load_images_dataset.PreparedData()
p_data.set_size((60, 25))
p_data.process_test_only()

yaml_file = open('C:\\Users\\grant\\Repos\\Cursive-OCR-for-Geneology\\slider_cnn0.8791080474853515_0.675.yaml', 'r')
loaded_model_yaml = yaml_file.read()
yaml_file.close()
model = model_from_yaml(loaded_model_yaml)
# load weights into new model
model.load_weights("C:\\Users\\grant\\Repos\\Cursive-OCR-for-Geneology\\model2_59.h5")
print("Loaded model from disk")
model.compile(loss='categorical_crossentropy',
              optimizer='Adadelta',
              metrics=['accuracy'])


score = model.evaluate(p_data.dataset['x_test'], p_data.dataset['y_test'], verbose=1)
test_loss = score[0]
test_accuracy = score[1]
print('Test accuracy:', test_accuracy)
print('Test loss:', test_loss)

pred = model.predict(p_data.dataset['x_test'])
print("predictions finished")
print(pred)
print(p_data.dataset['y_test'])
