import tensorflow as tf
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow import keras
import matplotlib.pyplot as plt

# Define the path to the CSV file
csv_file_path = r"D:\PHD_DOCS\Korea\Yonsei-vnl-coding-assignment-vision-48hrs\Yonsei-vnl-coding-assignment-vision-48hrs\dataset\data\cifar100_nl.csv"
test_csv = r"D:\PHD_DOCS\Korea\Yonsei-vnl-coding-assignment-vision-48hrs\Yonsei-vnl-coding-assignment-vision-48hrs\dataset\data\cifar100_nl_test.csv"

# Read the CSV file into a Pandas DataFrame
df = pd.read_csv(csv_file_path, header=None)
df_test = pd.read_csv(test_csv, header=None)

df.dropna(subset=[1], inplace=True)
df_test.dropna(subset=[1], inplace=True)


# Split the dataset into training and validation sets
train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)

# converting  class names to numeric labels
class_to_index = {class_name: index for index, class_name in enumerate(df[1].unique())}

train_df[1] = train_df[1].map(class_to_index)
val_df[1] = val_df[1].map(class_to_index)

df_test[1] = df_test[1].map(class_to_index)


# Define a function to load and preprocess images
def load_and_preprocess_image(image_path, label):
    image = tf.io.read_file(image_path)
    image = tf.image.decode_png(image, channels=3)
    image = tf.image.resize(image, (32, 32))  # input size
    image = tf.image.convert_image_dtype(
        image, tf.float32
    )  # Normalizing pixel values to 0,1
    return image, label


##Test
test_dataset = tf.data.Dataset.from_tensor_slices((df_test[0], df_test[1]))
test_dataset = test_dataset.map(load_and_preprocess_image)
test_dataset = test_dataset.batch(batch_size=64)
test_dataset = test_dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)


## Train
train_dataset = tf.data.Dataset.from_tensor_slices((train_df[0], train_df[1]))
train_dataset = train_dataset.map(load_and_preprocess_image)
train_dataset = train_dataset.shuffle(buffer_size=10000)
train_dataset = train_dataset.batch(batch_size=64)
train_dataset = train_dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

# Validation
val_dataset = tf.data.Dataset.from_tensor_slices((val_df[0], val_df[1]))
val_dataset = val_dataset.map(load_and_preprocess_image)
val_dataset = val_dataset.batch(batch_size=64)
val_dataset = val_dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)


## Defining Model

model = keras.Sequential(
    [
        keras.layers.Conv2D(32, (3, 3), activation="relu", input_shape=(32, 32, 3)),
        keras.layers.MaxPooling2D((2, 2)),
        keras.layers.Conv2D(64, (3, 3), activation="relu"),
        keras.layers.MaxPooling2D((2, 2)),
        keras.layers.Conv2D(128, (3, 3), activation="relu"),
        keras.layers.Flatten(),
        keras.layers.Dense(128, activation="relu"),
        keras.layers.Dense(100, activation="softmax"),
    ]
)


#  3 Convolutional layers (Conv2D) are responsible for feature extraction.
#  2 MaxPooling layers reduce spatial dimensions.
#  1 Flatten layer converts the 2D feature maps into a 1D vector for fully connected layers.
#  2 Fully connected layers are responsible for classification.


# Compile the model
model.compile(
    optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"]
)


history = model.fit(train_dataset, epochs=100, validation_data=val_dataset, verbose=1)


plt.plot(history.history["accuracy"], label="Training Accuracy")
plt.plot(history.history["val_accuracy"], label="Validation Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()

# Evaluate the model on the test dataset
test_loss, test_accuracy = model.evaluate(test_dataset)
print(f"Test Accuracy: {test_accuracy}")

plt.axhline(y=test_accuracy, color="r", linestyle="--", label="Test Accuracy")
plt.legend()
plt.show()
