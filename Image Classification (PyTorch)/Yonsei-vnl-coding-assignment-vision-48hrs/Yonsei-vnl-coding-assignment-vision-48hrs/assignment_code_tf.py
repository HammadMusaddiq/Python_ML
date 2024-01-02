import os
import math

import numpy as np
import pandas as pd

import keras
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D,Flatten,MaxPooling2D,Dropout,Dense,Activation,BatchNormalization
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import tensorflow.keras.layers as L
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers.schedules import ExponentialDecay
import tensorflow_hub as hub

from tqdm import tqdm
from tabulate import tabulate
from sklearn.model_selection import train_test_split, RepeatedStratifiedKFold


# class Dataset:
#     def __init__(self, data_root: str, *, test_size: float, img_size: int, seed: int = 0) -> None:
#         self.label2index = {}
#         self.index2label = {}
        
#         # Discover the class label names.
#         class_labels = os.listdir(data_root)
#         self.nclasses = len(class_labels)
#         X, y = [], []
        
#         for label_index, label in enumerate(class_labels):
#             # Load the images for this class label.
#             self.label2index[label_index] = label
#             self.index2label[label] = label_index
            
#             img_names = os.listdir(os.path.join(data_root, label))
#             for img_name in img_names:
#                 img_path = os.path.join(data_root, label, img_name)
#                 img = load_img(img_path, target_size=(img_size, img_size, 3))
#                 X.append(img_to_array(img))
#                 y.append(label_index)
        
#         X = np.array(X)
#         y = np.array(y)
#         one_hot_y = to_categorical(y, num_classes=self.nclasses)
        
#         # Make a stratified split.
#         self.X, self.X_test, self.labels, self.labels_test, self.y, self.y_test = train_test_split(
#             X, y, one_hot_y, test_size=test_size, random_state=seed, stratify=y)

# data = Dataset("dataset\data\cifar100_nl.csv", test_size=0.3, img_size=224)
# print(data.X.shape, data.y.shape)

data = pd.read_csv("dataset\data\cifar100_nl.csv", header = None)
data.dropna(subset=1, inplace=True)
data = data[:1000].copy()
image_paths = ["dataset/" + path for path in data[0].tolist()]
labels = data[1].tolist()

img_size = 224
index_label = {}

# Discover the class label names
class_labels = np.unique(labels)
nclasses = len(class_labels)
X, y, images = [], [], []

for label_index, label in enumerate(class_labels):
    # Load the images for this class label
    index_label[label_index] = label

    # Get image paths corresponding to this label
    label_image_paths = image_paths[labels == label]
    for img_path in label_image_paths:
        img = load_img(img_path, target_size=(img_size, img_size))
        images.append(img)
        X.append(img_to_array(img))
        y.append(label_index)

X = np.array(X)
y_old = np.array(y)
y = to_categorical(y, num_classes=nclasses)



class Dataset:
    def __init__(self, csv_file: str, *, test_size: float, img_size: int, seed: int = 0) -> None:
        self.label2index = {}
        self.index2label = {}
        
        # Read the CSV file to get image paths and labels.
        data = pd.read_csv(csv_file, header = None)
        data.dropna(subset=1, inplace=True)
        data = data[:1000].copy()
        image_paths = ["dataset/" + path for path in data[0].tolist()]
        labels = data[1].tolist()
        
        # Discover the class label names.
        class_labels = list(set(labels))
        self.nclasses = len(class_labels)
        X, y = [], []
        
        for label_index, label in enumerate(class_labels):
            self.label2index[label] = label_index
            self.index2label[label_index] = label
            
            # Get image paths corresponding to this label.
            label_image_paths = [image_paths[i] for i in range(len(image_paths)) if labels[i] == label]
            
            for img_path in label_image_paths:
                img = load_img(img_path, target_size=(img_size, img_size, 3))
                X.append(img_to_array(img))
                y.append(label_index)
        
        X = np.array(X)
        y = np.array(y)
        one_hot_y = to_categorical(y, num_classes=self.nclasses)
        
        # Make a stratified split.
        self.X, self.X_test, self.labels, self.labels_test, self.y, self.y_test = train_test_split(
            X, y, one_hot_y, test_size=test_size, random_state=seed, stratify=y)

# Provide the path to your CSV file containing image paths and labels.
csv_file_path = "dataset\data\cifar100_nl.csv"
data = Dataset(csv_file_path, test_size=0.3, img_size=224)
print(data.X.shape, data.y.shape)


embed = hub.KerasLayer("https://tfhub.dev/google/bit/m-r101x1/1", trainable=False)
X_embedding = embed(data.X)
X_test_embedding = embed(data.X_test)
print(X_embedding.shape, X_test_embedding.shape)

def make_model(
    nclasses: int, *, dropout_rate: float, nhiddenunits: int, l2_regularization: float
) -> tf.keras.Model:
    model = tf.keras.Sequential()
    # One fully connected hidden layer
    model.add(L.Dense(nhiddenunits, activation="relu", kernel_regularizer=l2(l2_regularization)))
    model.add(L.Dropout(dropout_rate))
    # Output layer
    model.add(L.Dense(nclasses, activation="softmax", kernel_regularizer=l2(l2_regularization)))
    return model

def evaluate_model(
    nclasses, X, y, X_test, y_test, *,
    epochs: int, batch_size: int, learning_rate: float,
    model_maker = make_model, **model_params
) -> tuple:
    
    # Math to compute the learning rate schedule. We will divide our
    # learning rate by a factor of 10 every 30% of the optimizer's
    # total steps.
    steps_per_epoch = math.ceil(len(X) / batch_size)
    third_of_total_steps = math.floor(epochs * steps_per_epoch / 3)
    
    # Make and compile the model.
    model = model_maker(nclasses, **model_params)
    model.compile(
        optimizer=Adam(
            learning_rate=ExponentialDecay(
                learning_rate,
                decay_steps=third_of_total_steps,
                decay_rate=0.1,
                staircase=True
            )
        ),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )
    
    # Train the model on the training set and evaluate it on the test set.
    history = model.fit(X, y, batch_size=batch_size, epochs=epochs, verbose=0)
    _, train_acc = model.evaluate(X, y, batch_size=batch_size, verbose=0)
    _, test_acc = model.evaluate(X_test, y_test, batch_size=batch_size, verbose=0)
    return model, train_acc, test_acc

def cv_evaluate_model(
    X, y, labels, *, nfolds: int, nrepeats: int, epochs: int, batch_size: int,
    learning_rate: float, model_maker, verbose: bool = True, seed: int = 0,
    **model_params
) -> dict:
    """
    Performs `nfolds` cross-validated training and evaluation of a
    model hyperparameter configuration. Returns a dictionary of
    statistics about the outcome of the cross-validated experiment.
    """
    _, nclasses = y.shape
    train_accs, test_accs = [], []
    
    # Train and evaluate the model for each fold.
    for train_index, test_index in tqdm(
        RepeatedStratifiedKFold(
            n_splits=nfolds, n_repeats=nrepeats, random_state=seed
        ).split(X, labels),
        total=nfolds*nrepeats, disable=not verbose
    ):
        
        # Select the data for this fold.
        X_train_fold = tf.gather(X, train_index) 
        y_train_fold = tf.gather(y, train_index)
        X_test_fold = tf.gather(X, test_index)
        y_test_fold = tf.gather(y, test_index)
        
        # Train and evaluate the model.
        _, train_acc, test_acc = evaluate_model(
            nclasses,
            X_train_fold,
            y_train_fold,
            X_test_fold,
            y_test_fold,
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            model_maker=model_maker,
            **model_params
        )
        train_accs.append(train_acc)
        test_accs.append(test_acc)
    
    # Aggregate.
    results = {
        "train_mean": np.mean(train_accs),
        "train_std": np.std(train_accs),
        "test_mean": np.mean(test_accs),
        "test_std": np.std(test_accs)
    }
    
    # Report.
    if verbose:
        print(
            tabulate(
                [
                    ["Train", results["train_mean"], results["train_std"]],
                    ["Test", results["test_mean"], results["test_std"]]
                ],
                headers=["Set", "Accuracy", "Standard Deviation"]
            )
        )
    
    return results


# We'll refer to these values throughout the notebook.
default_cv_evaluate_params = {
    "X": X_embedding,
    "y": data.y,
    "labels": data.labels,
    "nfolds": 10,
    "nrepeats": 3,
    "model_maker": make_model,
    "epochs": 200,
    "batch_size": 32,
    "verbose": False,
    "learning_rate": 3e-3, #0.003
    "dropout_rate": 0.3,
    "nhiddenunits": 64,
    "l2_regularization": 1e-6 #0.000001
}

_ = cv_evaluate_model(
    **{
        **default_cv_evaluate_params,
        "verbose": True
    }
)

X = X_embedding
y = data.y
labels = data.labels
train_accs, test_accs = [], []
history = []
    
# Train and evaluate the model for each fold.
for train_index, test_index in tqdm(
    RepeatedStratifiedKFold(
        n_splits=10, n_repeats=3, random_state=0
    ).split(X, labels),
    total=10*3, disable = not True #(verbose - True)
):

    # Select the data for this fold.
    X_train = tf.gather(X, train_index) 
    y_train = tf.gather(y, train_index)
    X_test = tf.gather(X, test_index)
    y_test = tf.gather(y, test_index)
    
    #Define Model
    model = tf.keras.Sequential()
    # One fully connected hidden layer
    model.add(L.Dense(64, activation="relu", kernel_regularizer=l2(0.000001)))
    model.add(L.Dropout(0.3))
    # Output layer
    model.add(L.Dense(17, activation="softmax", kernel_regularizer=l2(0.000001)))

    #Learning Rate
    steps_per_epoch = math.ceil(len(X_train) / 32) #batch - 32
    third_of_total_steps = math.floor(200 * steps_per_epoch / 3) #epoch - 200
    
    # Train and evaluate the model.
    model.compile(
        optimizer=Adam(
            learning_rate=ExponentialDecay(
                0.003,
                decay_steps=third_of_total_steps,
                decay_rate=0.1,
                staircase=True
            )
        ),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )
    
    # Train the model on the training set and evaluate it on the test set.
    history = (model.fit(X_train, y_train, batch_size=32, epochs=200, verbose=1, validation_data=(X_test, y_test)))
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