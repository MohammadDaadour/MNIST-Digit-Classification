from tensorflow import keras


model = keras.models.load_model("mnist_model.h5")
model.save("mnist_model.keras")
print("✅ mnist_model.h5 converted to mnist_model.keras")