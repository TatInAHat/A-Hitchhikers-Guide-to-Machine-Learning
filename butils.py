import csv
import numpy as np
import math

train = 'data/cumulative.csv'

# preprocess data


def get_data(filename):
    with open(filename, 'r') as data_file:
        reader = csv.reader(data_file, delimiter=',', quotechar='"')
        raw_data = [row for row in reader]
    raw_data = raw_data[54:]
    data_array = np.asarray(raw_data, dtype=str)

    columns = [4, 5, 9, 17, 21, 26, 30, 54, 59, 64]
    final = data_array[:, [3,11,26,29,32,38,44]]

    #print data_array


    print final


get_data(train)
