import time
import functools
from abc import abstractmethod

import numpy as np
import pandas as pd
from keras import layers, Input
from keras.layers import Dense, Dropout
from keras.models import Sequential
from keras.utils import to_categorical
from sklearn.datasets import fetch_openml
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


class MnistClassifierInterface:
  """
  Interface for the MNIST classifiers
  """
  MAX_PIXELS = 255.0
  NUM_CLASSES = 10  # number of values overall in MNIST dataset

  @abstractmethod
  def train(self, *args, **kwargs):
    """
    Run train data process
    """
    pass

  @abstractmethod
  def predict(self, *args, **kwargs):
    """
    Run predict data process
    """
    pass

  def __str__(self):
    return self.__class__.__name__


class RandomForest(MnistClassifierInterface):
  """
  A Random Forest classifier for MNIST image recognition tasks
  """
  def __init__(self, X_train, X_test, y_train, y_test, *args, **kwargs) -> None:
    self.X_train = X_train
    self.X_test = X_test
    self.y_train = y_train
    self.y_test = y_test
    self.model = RandomForestClassifier()

  def train(self, *args, **kwargs) -> float:
    """
    Train model and return accuracy score
    """
    self.model.fit(self.X_train, self.y_train)
    accuracy = self.model.score(self.X_test, self.y_test)
    return accuracy

  def predict(self) -> float:
    """
    Get model prediction results and return accuracy score
    """
    y_pred = self.model.predict(self.X_test)
    accuracy = accuracy_score(y_pred, self.y_test)
    return accuracy


class FeedForwardNeuralNetwork(MnistClassifierInterface):
  """
  A Feed-Forward Neural Network (FNN) classifier for MNIST image recognition tasks
  """
  def __init__(self, X_train: pd.DataFrame, X_test: pd.DataFrame,
               y_train: pd.DataFrame, y_test: pd.DataFrame,
               epochs: int=5) -> None:
    self.X_train = self._normalize_input(X_train)
    self.X_test = self._normalize_input(X_test)
    self.y_train = self._convert_to_categorical(y_train)
    self.y_test = self._convert_to_categorical(y_test)

    self.epochs = epochs
    self.model = self._get_model()

  def _normalize_input(self, data: pd.DataFrame) -> np.ndarray:
    """
    Normalize input data. Applicable for X_train, X_test.
    No need to reshape for FNN, since we already get data as one long vector 784.
    """
    normalized = data / self.MAX_PIXELS
    return normalized

  def _convert_to_categorical(self, data: pd.DataFrame) -> np.ndarray:
    """
    Convert labels into one-hot encoded vectors. Applicable for y_train, y_test.
    """
    encoded = to_categorical(data, self.NUM_CLASSES)
    return encoded

  def _get_model(self) -> Sequential:
    """
    Create model for the classifier
    """
    model = Sequential()
    model.add(Input(shape=(self.X_train.shape[1],)))
    model.add(Dense(units=128, activation='relu'))
    model.add(Dense(units=64, activation='relu'))
    model.add(Dropout(0.25))  # reduce overfitting
    model.add(Dense(units=self.NUM_CLASSES, activation='softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

  def train(self, batch_size=256) -> float:
    """
    Train model and return accuracy score
    """
    self.model.fit(self.X_train, self.y_train, batch_size=batch_size, epochs=self.epochs)
    _, accuracy = self.model.evaluate(self.X_test, self.y_test, batch_size=batch_size)
    return accuracy

  def predict(self) -> float:
    """
    Get model prediction results and return accuracy score
    """
    y_pred_probs = self.model.predict(self.X_test)
    y_pred = np.argmax(y_pred_probs, axis=1)
    accuracy = accuracy_score(y_pred, np.argmax(self.y_test, axis=1))
    return accuracy


class ConvolutionalNeuralNetwork(MnistClassifierInterface):
  """
  A Convolutional Neural Network (CNN) classifier for MNIST image recognition tasks
  """
  IMG_SIZE = (28, 28)  # width and height of the image

  def __init__(self, X_train: pd.DataFrame, X_test: pd.DataFrame,
               y_train: pd.DataFrame, y_test: pd.DataFrame,
               epochs: int=5) -> None:
    self.X_train = self._normalize_input(X_train)
    self.X_test = self._normalize_input(X_test)
    self.y_train = self._convert_to_categorical(y_train)
    self.y_test = self._convert_to_categorical(y_test)

    self.epochs = epochs
    self.model = self._get_model()

  def _normalize_input(self, data: pd.DataFrame) -> np.ndarray:
    """
    Normalize input data. Applicable for X_train, X_test.
    Need to reshape for CNN, since we get data as one long vector 784, but we need 28*28
    """
    normalized = data / self.MAX_PIXELS
    res = normalized.values.reshape(-1, *self.IMG_SIZE)
    return res

  def _convert_to_categorical(self, data: pd.DataFrame) -> np.ndarray:
    """
    Convert labels into one-hot encoded vectors. Applicable for y_train, y_test.
    """
    encoded = to_categorical(data, self.NUM_CLASSES)
    return encoded

  def _get_model(self) -> Sequential:
    """
    Create model for the classifier
    """
    model = Sequential()
    model.add(Input(shape=(*self.IMG_SIZE, 1)))
    model.add(layers.Conv2D(32, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Flatten())
    model.add(layers.Dense(128, activation='relu'))
    model.add(Dropout(0.25))
    model.add(layers.Dense(self.NUM_CLASSES, activation='softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

  def train(self, batch_size=256) -> float:
    """
    Train model and return accuracy score
    """
    self.model.fit(self.X_train, self.y_train, batch_size=batch_size, epochs=self.epochs)
    _, accuracy = self.model.evaluate(self.X_test, self.y_test, batch_size=batch_size)
    return accuracy

  def predict(self) -> float:
    """
    Get model prediction results and return accuracy score
    """
    y_pred_probs = self.model.predict(self.X_test)
    y_pred = np.argmax(y_pred_probs, axis=1)
    accuracy = accuracy_score(y_pred, np.argmax(self.y_test, axis=1))
    return accuracy


class MnistClassifier:
  """
  Factory class to instantiate requested model
  """
  @functools.cached_property
  def mnist_data(self) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Cached property with MNIST data.
    Object can be reused for different classifiers w/o reloading MNIST data.
    """
    mnist = fetch_openml('mnist_784')
    X_train, X_test, y_train, y_test = train_test_split(mnist.data, mnist.target, test_size=1/7)  # setup for test data to be 1000 out of 7000
    return X_train, X_test, y_train, y_test

  def get_model(self, code: str) -> MnistClassifierInterface:
    """
    Instantiate requested classifier
    """
    classifiers = {
        'cnn': ConvolutionalNeuralNetwork,
        'rf': RandomForest,
        'nn': FeedForwardNeuralNetwork,
    }
    if code in classifiers:
      # X_train, X_test, y_train, y_test = self.mnist_data
      return classifiers[code](*self.mnist_data)

    raise ValueError(f'Got unexpected value: {code}. Expected values: {list(classifiers)}')


def run_models():
  """
  Run training and prediction for Random Forest, Feed-Forward Neural Network and Convolutional Neural Network models
  and print results. When classifier created, it will read data on instantiation and later reuse cached data
  as input for all 3 models (this improves performance, avoids re-loading data anew for each model).
  """
  classifier = MnistClassifier()
  algorithm_list = ['cnn', 'rf', 'nn']
  acc_predict_model = {}

  for code in algorithm_list:
    start = time.perf_counter()

    model = classifier.get_model(code)
    print(f'Called {str(model)}')
    acc_train = model.train()
    acc_predict = model.predict()
    acc_predict_model[str(model)] = acc_predict
    print(f'>>> Training accuracy {acc_train}. Prediction accuracy {acc_predict}')

    minutes, sec = divmod(time.perf_counter() - start, 60)
    print(f'Time elapsed {minutes:0f} min {sec:.0f} sec')

  sorted_acc = dict(sorted(acc_predict_model.items(), key=lambda item: item[1], reverse=True))
  print(f'Accuracy score sorted by model: {sorted_acc} min')


run_models()
