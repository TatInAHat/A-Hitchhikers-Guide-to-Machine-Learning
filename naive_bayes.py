import csv
import numpy as np
import math
import tools
from sklearn.naive_bayes import GaussianNB

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

    gnb = GaussianNB()
    # y_pred = gnb.fit(train_points, train_labels).predict(train_points)
    # y_pred2 = gnb.fit(train_points, train_labels).predict(test_points)

    # earth = [365.25, 1, 255, 1., 5777, 1]
    # earth = np.asarray(earth)

    jupiter = [4300, 11.209, 109.9, 0.05026, 5777, 1]
    jupiter = np.asarray(jupiter)

    print gnb.fit(train_points, train_labels).predict(jupiter.reshape(1,-1))
    # print("Number of mislabeled training points out of a total %d points : %d" % (train_points.shape[0],(train_labels != y_pred).sum()))
    # print("Number of mislabeled testing points out of a total %d points : %d" % (test_points.shape[0],(test_labels != y_pred2).sum()))

main()
