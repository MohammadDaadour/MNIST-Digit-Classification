import numpy as np
from tensorflow import keras
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt

# Load data
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

# Your original scaling
x_train = (x_train / 255.0) * 9
x_test = (x_test / 255.0) * 9

# Flatten images
x_train = x_train.reshape(60000, -1)
x_test = x_test.reshape(10000, -1)

# Build model
model = keras.Sequential([
    keras.layers.Dense(128, activation='relu', input_shape=(784,)),
    keras.layers.Dense(10, activation='softmax')
])

model.compile(
    loss='sparse_categorical_crossentropy',
    optimizer='sgd',
    metrics=['accuracy']
)




early_stop = EarlyStopping(
    monitor='val_loss',
    patience=3,
    restore_best_weights=True
)




history = model.fit(
    x_train, y_train,
    epochs=20,
    validation_data=(x_test, y_test),
    callbacks=[early_stop]
)


model.save('mnist_model.h5')


y_pred = np.argmax(model.predict(x_test), axis=1)
print(classification_report(y_test, y_pred))




