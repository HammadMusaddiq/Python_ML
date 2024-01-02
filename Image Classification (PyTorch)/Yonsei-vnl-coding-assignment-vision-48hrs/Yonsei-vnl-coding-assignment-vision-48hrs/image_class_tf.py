import os
import math

import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import tensorflow.keras.layers as L
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers.schedules import ExponentialDecay
import tensorflow_hub as hub

from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D,Flatten,MaxPooling2D,Dropout,Dense,Activation,BatchNormalization

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, RepeatedStratifiedKFold

from tqdm import tqdm
from tabulate import tabulate


data = pd.read_csv("dataset\data\cifar100_nl.csv", header = None)
data.dropna(subset=1, inplace=True)
labels = data[1].tolist()

img_size = 15

# Discover the class label names
class_labels = np.unique(labels)
nclasses = len(class_labels)
X, y, images = [], [], []

print(len(class_labels))

for label_index, label in enumerate(class_labels):

    # Get image paths corresponding to this label
    label_image_paths = data[data[1] == label][0].tolist()
    # label_image_paths = image_paths[labels == label]
    for img_path in label_image_paths:
        img_path = "dataset/" + img_path
        img = load_img(img_path, target_size=(img_size, img_size))
        images.append(img)
        X.append(img_to_array(img))
        y.append(label_index)


X = np.array(X)
y_old = np.array(y)
y = to_categorical(y, num_classes=nclasses)


# data = pd.read_csv("dataset\data\cifar100_nl.csv", header = None)
# data.dropna(subset=1, inplace=True)
# # data = data[:1000].copy()
# image_paths = ["dataset/" + path for path in data[0].tolist()]
# labels = data[1].tolist()

# img_size = 15
# index_label = {}

# # Discover the class label names
# class_labels = np.unique(labels)
# nclasses = len(class_labels)

# class_dict = {}
# for label_index, label in enumerate(class_labels):
#     class_dict[label_index] = label

# X, y, images = [], [], []


# for img_path, label in zip(image_paths,labels):
    
#     img = load_img(img_path, target_size=(img_size, img_size))
#     images.append(img)
#     X.append(img_to_array(img))

#     for key, value in class_dict.items():
#         if value == label:
#             y.append(key)
#             break


print(X.shape)
print(y.shape)


labels = y_old
train_accs, test_accs = [], []
history = []
verbose = True
i = 0


# RepeatedStratifiedKFold(
#         n_splits=10, n_repeats=3, random_state=0
#     ).split(X, labels)

# Train and evaluate the model for each fold.
for train_index, test_index in tqdm(RepeatedStratifiedKFold(
        n_splits=10, n_repeats=3, random_state=0
    ).split(X, labels), total=10*3, disable = not True): #(verbose - True)

    # Select the data for this fold.
    X_train = tf.gather(X, train_index) 
    y_train = tf.gather(y, train_index)
    X_test = tf.gather(X, test_index)
    y_test = tf.gather(y, test_index)

    if (i==0):
        i=1
        print(X_train.shape)
        print(X_test.shape)
        print(y_train.shape)
        print(y_test.shape)
    
    #Define Model
    #Define Model
    model = Sequential()
    model.add(Conv2D(8, (3,3), input_shape=(15,15,3), kernel_initializer='he_uniform', activation="relu"))
    model.add(MaxPooling2D(2,2))
    model.add(Conv2D(16, (3,3), kernel_initializer='he_uniform', activation="relu"))
    model.add(MaxPooling2D(2,2))
    model.add(Flatten())
    model.add(Dense(100,activation="relu"))
    model.add(Dense(nclasses, activation="softmax"))

    #Learning Rate
    steps_per_epoch = math.ceil(len(X_train) / 32) #batch - 32
    third_of_total_steps = math.floor(10 * steps_per_epoch / 3) #epoch - 10
    
    # Train and evaluate the model.
    model.compile(
        optimizer=Adam(
            learning_rate=ExponentialDecay(
                0.0003,
                decay_steps=third_of_total_steps,
                decay_rate=0.1,
                staircase=True
            )
        ),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )
    
    # Train the model on the training set and evaluate it on the test set.
    history = (model.fit(X_train, y_train, batch_size=32, epochs=10, verbose=0, validation_data=(X_test, y_test)))
    train_loss, train_acc = model.evaluate(X_train, y_train, batch_size=32, verbose=0)




    test_loss, test_acc = model.evaluate(X_test, y_test, batch_size=32, verbose=0)
    
    train_accs.append(train_acc)
    test_accs.append(test_acc)

# Aggregate.
results = {
    "Train_Acc": np.mean(train_accs),
    "Train_std": np.std(train_accs),
    "Test_Acc": np.mean(test_accs),
    "Test_std": np.std(test_accs)
}

# Report.
if verbose:
    print(
        tabulate(
            [
                ["Train", results["Train_Acc"], results["Train_std"]],
                ["Test", results["Test_Acc"], results["Test_std"]]
            ],
            headers=["Set", "Accuracy", "Standard Deviation"]
        )
    )


model.save('image_cifer100.h5')