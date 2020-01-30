import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from time import time

from sklearn.model_selection import train_test_split
import tensorflow as tf
import keras
import keras.layers as layers
from keras.regularizers import l2
from keras.models import Sequential
from keras.preprocessing.image import ImageDataGenerator
from keras.utils.np_utils import to_categorical
from keras.callbacks import TensorBoard
from keras.datasets import mnist
from datagenerator import DataGenerator, generate_generator_objects

# Load mnist data
(X_train_raw, y_train_raw), (X_test, y_test) = mnist.load_data()
X_train_raw = X_train_raw.reshape(len(X_train_raw), 28, 28, 1)

# Prepare and set aside the test set
X_test = X_test.reshape(len(X_test), 28, 28, 1)

sns.set()

params = {'dim': (32,32),
          'batch_size': 256,
          'n_classes': 10,
          'n_channels': 1,
          'shuffle': True}


def generate_lenet5_model(activation='relu', kernel_regularizer=None) -> keras.Sequential:
    # Architecture from picture at: https://www.researchgate.net/figure/Architecture-of-LeNet-5_fig3_313808170
    model = keras.Sequential()

    # C1: features maps
    model.add(layers.Conv2D(filters=6, kernel_size=(3, 3), activation=activation, input_shape=(32, 32, 1), kernel_regularizer=kernel_regularizer))

    # S2: Feature maps
    model.add(layers.AveragePooling2D())

    # C3: Feature maps
    model.add(layers.Conv2D(filters=16, kernel_size=(3, 3), activation=activation, kernel_regularizer=kernel_regularizer))

    # S4: Feature maps
    model.add(layers.AveragePooling2D())
    
    model.add(layers.Flatten())

    # C5: Layer
    model.add(layers.Dense(units=120, activation=activation, kernel_regularizer=kernel_regularizer))

    # F6: Layer
    model.add(layers.Dense(units=84, activation=activation, kernel_regularizer=kernel_regularizer))

    # 7: Output layer
    model.add(layers.Dense(units=10, activation='softmax', kernel_regularizer=kernel_regularizer))
    print(model.summary())

    model.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adam(), metrics=['accuracy'])

    return model

def prepare_data(X: np.ndarray) -> np.ndarray:
    """ Pad a 28x28 picture into a 32x32 picture """
    X_new = np.pad(X, ((0, 0), (2, 2), (2, 2), (0, 0)), 'constant')
    return X_new

""" Given a keras model, train the model with given parameters """
def train_keras_model(X_train: np.ndarray, y_train: np.ndarray, model: keras.Sequential, EPOCHS: int, BATCH_SIZE: int, generator: bool) -> tf.keras.callbacks.History:
    # Randomly split into a train and validation set
    X_train, X_validation, y_train, y_validation = train_test_split(X_train, y_train, test_size=0.2)
    X_train, X_validation = prepare_data(X_train), prepare_data(X_validation)

    y_train, y_validation = to_categorical(y_train), to_categorical(y_validation)

    if not generator:
        return model.fit(X_train, y_train, batch_size=BATCH_SIZE, epochs=EPOCHS, validation_data=(X_validation, y_validation), shuffle=True)
    (partition, labels) = generate_generator_objects()
    train_generator = DataGenerator(partition['train'], labels, **params)
    validation_generator = DataGenerator(partition['validation'], labels, **params)
    
    return model.fit_generator(train_generator, validation_data=validation_generator, epochs=EPOCHS)

""" Plot loss from keras history object """
def plot_history_loss(history: tf.keras.callbacks.History, name: str):
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Loss per epoch for ' + name)
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.show()

""" Plot accuracy from keras history object """
def plot_history_accuracy(history: tf.keras.callbacks.History, name: str):
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Accuracy per epoch for ' + name)
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.show()

if __name__ == "__main__":
    """ Hyper paramters """
    activation_functions = [("relu", "ReLU"), ("sigmoid", "Sigmoid")]
    kernel_regularizers = [(None, "No regularization"), (l2(0.001), "L2: 0.001"), (l2(0.01), "L2: 0.01"), (l2(0.1), "L2: 0.1"), (l2(1), "L2: 1")]

    """ Training parameters """
    EPOCHS = 5
    BATCH_SIZE = 256

    """ Initialize paramters for selecting best hyper parameters """
    best_accuracy = 0
    best_accuracy_params = None
    best_accuracy_name = ""


    """ For each pair of hyper paramters, try to do 
        a train/validation split, and train with training data,
        and check performance with the validation data.
        Selects the model with the highest accuracy """
    for (activation_func, activation_func_displayname) in activation_functions:
        for (regularizer, regularizer_displayname) in kernel_regularizers:
            model = generate_lenet5_model(activation=activation_func, kernel_regularizer=regularizer)
            history = train_keras_model(X_train_raw, y_train_raw, model, EPOCHS, BATCH_SIZE, True)
            name = "%s - %s" % (activation_func_displayname, regularizer_displayname)
            plot_history_loss(history, name)
            plot_history_accuracy(history, name)
            if history.history["accuracy"][-1] > best_accuracy:
                best_accuracy = history.history["accuracy"][-1]
                best_accuracy_params = (activation_func, regularizer)
                best_accuracy_name = name
    
    model = generate_lenet5_model(activation=best_accuracy_params[0], kernel_regularizer=best_accuracy_params[1])

    """ Using the test set that we set aside earlier, evaluate performance on the selected best model """
    X_train, y_train = prepare_data(X_train_raw), to_categorical(y_train_raw)
    X_test, y_test = prepare_data(X_test), to_categorical(y_test)
    
    history = model.fit(x=X_train, y=y_train, validation_data=(X_test, y_test),batch_size=BATCH_SIZE, epochs=EPOCHS)
    print("Test accuracy: %r" % (history.history["val_accuracy"][-1]))
    
