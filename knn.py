import csv
import numpy as np
from sklearn.neighbors import NearestNeighbors
import tools

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

    nbrs = NearestNeighbors(n_neighbors=2, algorithm='ball_tree').fit(train_points)
    distances, indices = nbrs.kneighbors(train_points)

    return 0


main()
