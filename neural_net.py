import tensorflow as tf
import numpy as np
import keras
from keras.layers.core import Dense, Dropout
import tools

tf.python.control_flow_ops = tf


def parameter_search(hidden_layers, X_train, y_train, X_valid, y_valid,
                   n_epochs=10, loss='mean_squared_error', optimizer='adam'):
    s

    return 0


def main():
    file = 'data/cumulative.csv'
    X = tools.get_data(file)
    esi = tools.ESI(X)
    X = tools.classify(X, esi)

    train, test = tools.train_test(X)

    train_points = train[:, :6]
    train_labels = train[:, 6]
    test_points = test[:, :6]
    test_labels = test[:, 6]


    return 0

main()
