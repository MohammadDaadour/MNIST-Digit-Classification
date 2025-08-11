import numpy as np
from tensorflow import keras
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt

# Load data
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

# Reshape for ImageDataGenerator (keep channels)
x_train = np.expand_dims(x_train, -1) / 255.0  # (60000, 28, 28, 1)
x_test = np.expand_dims(x_test, -1) / 255.0    # (10000, 28, 28, 1)

# Create augmentation generator
datagen = ImageDataGenerator(
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.1,
    shear_range=0.1
)
datagen.fit(x_train)

# Build simple dense model (will flatten inside model)
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28, 1)),
    keras.layers.Dense(256, activation='relu'),
    keras.layers.Dense(10, activation='softmax')
])

model.compile(
    loss='sparse_categorical_crossentropy',
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    metrics=['accuracy']
)

# Early stopping
early_stop = EarlyStopping(
    monitor='val_loss',
    patience=10,
    restore_best_weights=True
)

# Train using augmented data
history = model.fit(
    datagen.flow(x_train, y_train, batch_size=128),
    validation_data=(x_test, y_test),
    epochs=50,
    callbacks=[early_stop]
)

# Save model
model.save('mnist_model_aug.h5')

# Evaluate
y_pred = np.argmax(model.predict(x_test), axis=1)
print(classification_report(y_test, y_pred))

# Plot history
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.legend()
plt.title("Loss")

plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.legend()
plt.title("Accuracy")

plt.show()

# Final accuracy
test_loss, test_accuracy = model.evaluate(x_test, y_test)
print(f"Final Test Accuracy: {test_accuracy:.4f}")
