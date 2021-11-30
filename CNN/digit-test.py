from PIL import Image, ImageOps
import numpy as np
from tensorflow import keras

model = keras.models.load_model("mnist_cnn_class.h5", compile = True)
model.summary()

x_test = []
y_test = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
y_test = keras.utils.to_categorical(y_test, 10)

for i in range(0, 10, 1):
    img = ImageOps.grayscale(Image.open('digits/{}.png'.format(i)))
    img.show()
    img_array = np.array(img).astype("float32") / 255
    img_array[img_array==1] = 0
    img_array = np.expand_dims(img_array, -1)
    x_test.append(img_array)

x_test = np.array(x_test)

score = model.evaluate(x_test, y_test, verbose=0)
print("Test loss:", score[0])
print("Test accuracy:", score[1])

predictions = model.predict(x_test)
classes = np.argmax(predictions, axis = 1)

print(classes)