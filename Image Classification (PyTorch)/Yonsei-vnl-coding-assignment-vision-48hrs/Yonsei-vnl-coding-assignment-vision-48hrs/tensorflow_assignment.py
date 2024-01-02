import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# Load and preprocess the training data
def preprocess_image(image_path, label):
    image = tf.io.read_file(image_path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, (224, 224))
    image = tf.cast(image, tf.float32) / 255.0
    return image, label

# Load the CSV file into a Pandas DataFrame
train_data = pd.read_csv("dataset\data\cifar100_nl.csv", header=None)
train_data.dropna(subset=1, inplace=True)
train_data[0] = "dataset/" + train_data[0].astype(str)
train_image_paths = train_data[0].values
train_labels = train_data[1].values

unique_labels = train_data[1].unique()
num_classes = len(unique_labels)

train_dataset = tf.data.Dataset.from_tensor_slices((train_image_paths, train_labels))
train_dataset = train_dataset.map(preprocess_image)
train_dataset = train_dataset.shuffle(buffer_size=len(train_data)).batch(32)

model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(num_classes, activation='softmax')
])

# # Compute class weights
# class_labels = np.unique(train_labels)
# class_counts = np.bincount(train_labels)
# total_samples = len(train_labels)
# class_weights = total_samples / (len(class_labels) * class_counts)

# # Create a dictionary to map class indices to their respective weights
# class_weights_dict = dict(enumerate(class_weights))

# # Use the class weights when compiling your model
# model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model without specifying class weights
# model.fit(trainX, trainY, epochs=35, batch_size=64, verbose=2, validation_data=(testX, testY))
model.fit(train_dataset, epochs=10)

# Train the model with class weights
# model.fit(train_dataset, epochs=10, class_weight=class_weights_dict)


# test data
test_data = pd.read_csv("dataset\data\cifar100_nl_test.csv", header=None)
test_data.dropna(subset=1, inplace=True)
test_data[0] = "dataset/" + test_data[0].astype(str)
test_image_paths = test_data[0].values
test_labels = test_data[1].values

test_dataset = tf.data.Dataset.from_tensor_slices((test_image_paths, test_labels))
test_dataset = test_dataset.map(preprocess_image)
test_dataset = test_dataset.batch(32)

# Make predictions on the test data
predictions = model.predict(test_dataset)

# Assuming you have the true labels for the test data, you can evaluate the model
true_labels = to_categorical(test_labels, num_classes=num_classes)  # Convert labels to one-hot encoding if not already
accuracy = np.mean(np.argmax(predictions, axis=1) == np.argmax(true_labels, axis=1))
print(f"Test Accuracy: {accuracy}")

# You can also generate a classification report
report = classification_report(np.argmax(true_labels, axis=1), np.argmax(predictions, axis=1))
print(report)
