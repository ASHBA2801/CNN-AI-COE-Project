import tensorflow as tf
from tensorflow.keras import layers, models

# Load the dog and cat dataset
train_data = tf.keras.utils.image_dataset_from_directory(
    "PetImages",
    image_size=(128, 128),
    batch_size=32,
    validation_split=0.2,
    subset="training",
    seed=123
)

val_data = tf.keras.utils.image_dataset_from_directory(
    "PetImages",
    image_size=(128, 128),
    batch_size=32,
    validation_split=0.2,
    subset="validation",
    seed=123
)

# Build CNN with three convolutional layers
model = models.Sequential([
    layers.Rescaling(1./255, input_shape=(128, 128, 3)),

    layers.Conv2D(32, (3, 3), activation="relu"),
    layers.MaxPooling2D((2, 2)),

    layers.Conv2D(64, (3, 3), activation="relu"),
    layers.MaxPooling2D((2, 2)),

    layers.Conv2D(128, (3, 3), activation="relu"),
    layers.MaxPooling2D((2, 2)),

    layers.Flatten(),
    layers.Dense(128, activation="relu"),
    layers.Dense(1, activation="sigmoid")
])

# Compile the model
model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

# Train the model
model.fit(train_data, validation_data=val_data, epochs=10)

# Evaluate the model
loss, accuracy = model.evaluate(val_data)
print("Validation Accuracy:", accuracy * 100, "%")

# Predict an image
img = tf.keras.utils.load_img(
    "test_cat.jpg",
    target_size=(128, 128)
)

img_array = tf.keras.utils.img_to_array(img)
img_array = tf.expand_dims(img_array, 0)

prediction = model.predict(img_array)

print("Dog" if prediction[0][0] >= 0.5 else "Cat")