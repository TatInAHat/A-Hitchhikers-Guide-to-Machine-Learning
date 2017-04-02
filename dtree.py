import csv
import numpy as np
from sklearn import tree
import tools


def main():
    file = 'data/cumulative.csv'
    means, stdevs, final = tools.get_data(file)
    esi = tools.ESI(final)

    classed = tools.classify(final, esi)

    train, test = tools.train_test(classed)

    train_points = train[:, :6]
    train_labels = train[:, 6]
    test_points = test[:, :6]
    test_labels = test[:, 6]

    clf = tree.DecisionTreeClassifier()
    clf.fit(train_points, train_labels)

    b = clf.predict(train_points)

    a = clf.predict(test_points)
    # Ein = 0.0

    # for j in xrange(len(train_points)):
    #     if b[j] != train_labels[j]:
    #         Ein += 1

    # Eout = 0.0

    # for i in xrange(len(test_points)):
    #     if a[i] != test_labels[i]:
    #         Eout += 1

    # earth = [365.25, 1, 255, 1., 5777, 1]
    # earth = np.asarray(earth)

    # jupiter = [4300, 11.209, 109.9, 0.05026, 5777, 1]
    # jupiter = np.asarray(jupiter)
    # print clf.predict(jupiter)
    return 0


main()