import numpy as np
from tensorflow import keras
from tensorflow.keras import layers


num_classes = 10
input_shape = (28, 28, 1)

# -----MNIST DATA-----
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

x_train = x_train.astype("float32") / 255
x_test = x_test.astype("float32") / 255

x_train = np.expand_dims(x_train, -1)
x_test = np.expand_dims(x_test, -1)
print(x_train.shape, "train shape")
print(x_test.shape, "test shape")


y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)
print(y_train.shape, "y_train shape")
print(y_test.shape, "y_test shape")
# -----MNIST DATA-----

# -----MODEL-----
model = keras.Sequential(
    [
        keras.Input(shape=input_shape),
        layers.Conv2D(16, kernel_size=(3, 3), activation="relu"),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Conv2D(16, kernel_size=(3, 3), activation="relu"),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Conv2D(16, kernel_size=(3, 3), activation="relu"),
        layers.Flatten(),
        layers.Dense(16, activation="relu"),
        layers.Dense(num_classes, activation="softmax"),
    ]
)
model.summary()
# -----MODEL-----


# -----MODEL FITTING-----
batch_size = 128
epochs = 10
model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_split=0.1)
model.save('mnist_cnn_class.h5')
# -----MODEL FITTING-----

model = keras.models.load_model("mnist_cnn_class.h5")
model.summary()

# -----MODEL LOSS\ACCURACY-----
score = model.evaluate(x_test, y_test, verbose=0)
print("Test loss:", score[0])
print("Test accuracy:", score[1])
# -----MODEL LOSS\ACCURACY-----
