import custom_models
import load_images_dataset

p_data = load_images_dataset.PreparedData()
p_data.process()

epochs = 20
batch_size = 64

model = custom_models.cursive_cnn(p_data.dataset['x_train'], p_data.dataset['y_train'],
                                    p_data.dataset['x_val'], p_data.dataset['y_val'],
                                    p_data.size, p_data.n_classes)

score = model.evaluate(x_test, y_test, verbose=1)

print('Test loss:', score[0])
print('Test accuracy:', score[1])
print(score)


pred = model.predict(p_data.dataset['x_test'])
print("predictions finished")
print(pred)
print(p_data.dataset['y_test'])
