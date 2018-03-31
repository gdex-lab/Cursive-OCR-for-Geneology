import custom_models
import load_images_dataset
from keras.models import model_from_yaml

p_data = load_images_dataset.PreparedData()
# p_data.set_size((60, 25))
p_data.process()

print(p_data.dataset['y_val'][:7])

epochs = 300
batch_size = 64

model = custom_models.cursive_cnn(p_data.dataset['x_train'], p_data.dataset['y_train'],
                                    p_data.dataset['x_val'], p_data.dataset['y_val'],
                                    p_data.size, p_data.n_classes,
                                    epochs=epochs, batch_size=batch_size)

score = model.evaluate(p_data.dataset['x_test'], p_data.dataset['y_test'], verbose=1)

# serialize model to YAML
model_yaml = model.to_yaml()
with open("C:\\Users\\grant\\Repos\\Cursive-OCR-for-Geneology\\slider_cnn.yaml", "w") as yaml_file:
    yaml_file.write(model_yaml)
# serialize weights to HDF5
model.save_weights("model.h5")
print("Saved model to disk")

# later...

# load YAML and create model
yaml_file = open('C:\\Users\\grant\\Repos\\Cursive-OCR-for-Geneology\\slider_cnn.yaml', 'r')
loaded_model_yaml = yaml_file.read()
yaml_file.close()
loaded_model = model_from_yaml(loaded_model_yaml)
# load weights into new model
loaded_model.load_weights("model.h5")
print("Loaded model from disk")


print('Test loss:', score[0])
print('Test accuracy:', score[1])
print(score)


pred = loaded_model.predict(p_data.dataset['x_test'])
print("predictions finished")
print(pred)
print(p_data.dataset['y_test'])
