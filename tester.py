from keras.models import model_from_yaml
import load_images_dataset
import numpy as np
import operator

p_data = load_images_dataset.PreparedData()
p_data.set_size((60, 25))
p_data.process_test_only()

yaml_file = open('C:\\Users\\grant\\Repos\\Cursive-OCR-for-Geneology\\v4_cnn.yaml', 'r')

loaded_model_yaml = yaml_file.read()
yaml_file.close()
model = model_from_yaml(loaded_model_yaml)


model.load_weights("C:\\Users\\grant\\Repos\\Cursive-OCR-for-Geneology\\v5_model_0.8.h5")
model.compile(
loss='categorical_crossentropy',
              optimizer='Adadelta',
              metrics=['accuracy'])


# score = model.evaluate(p_data.dataset['x_test'], p_data.dataset['y_test'], verbose=1)
# test_loss = score[0]
# test_accuracy = score[1]
# print('Test accuracy:', test_accuracy)
# print('Test loss:', test_loss)
# print(p_data.dataset['x_test'])
# print(p_data.dataset['y_test'])

pred = model.predict(p_data.dataset['x_test'])
# print(pred)
print("predictions finished")

vowels = ['a', 'e', 'i', 'o', 'u']
from queue import Queue
clean_actuals = Queue()
for d in p_data.dataset['y_test']:
    clean_actuals.put(vowels[np.where(d == 1)[0][0]])

for p in pred:
    prediction_dict = {'a': p[0],
                        'e': p[1],
                        'i': p[2],
                        'o': p[3],
                        'u': p[4],
                         }
    sorted_preds = sorted(prediction_dict.items(), key=operator.itemgetter(1), reverse=True)
    print("-------------------")
    print("Actual: {}".format(clean_actuals.get()))
    print("Predicted: {}".format(sorted_preds))
    print("-------------------")

# print(pred)
# print(p_data.dataset['y_test'])
