import csv
import numpy as np

train = 'data/planets.csv'

# preprocess data

def get_data(filename):
    with open(filename, 'r') as data_file:
        reader = csv.reader(data_file, delimiter=',', quotechar='"')
        raw_data = [row for row in reader]
    raw_data = raw_data[74:]
    data_array = np.asarray(raw_data, dtype=str)

    columns = [4,5,9,13,17,21,26,30,54,59,64]
    final = data_array[:,0]

    for i in xrange(len(columns)):
        final = np.c_[final, data_array[:,columns[i]]]

    print final
    return final

get_data(train)
