import csv
import numpy as np
import math
import tools
from sklearn.naive_bayes import GaussianNB

def main(parameters):
    file = 'data/cumulative.csv'
    X = tools.get_data(file)
    esi = tools.ESI(X)
    X = tools.classify(X, esi)

    train, test = tools.train_test(X)

    train_points = train[:, :6]
    train_labels = train[:, 6]
    test_points = test[:, :6]
    test_labels = test[:, 6]

    gnb = GaussianNB()
    # y_pred = gnb.fit(train_points, train_labels).predict(train_points)
    # y_pred2 = gnb.fit(train_points, train_labels).predict(test_points)

    parameters = np.asarray(parameters)

    return gnb.fit(train_points, train_labels).predict(parameters.reshape(1,-1))
