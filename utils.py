import csv
import numpy as np
import math

train = 'data/planets.csv'

# preprocess data


def get_data(filename):
    with open(filename, 'r') as data_file:
        reader = csv.reader(data_file, delimiter=',', quotechar='"')
        raw_data = [row for row in reader]
    raw_data = raw_data[74:]
    data_array = np.asarray(raw_data, dtype=str)

    columns = [4, 5, 9, 17, 21, 26, 30, 54, 59, 64]
    final = data_array[:, 0]

    for i in xrange(len(columns)):
        final = np.c_[final, data_array[:, columns[i]]]

    t1 = np.nan
    t2 = str(t1)

    final[final == ''] = t2

    final = final.astype(float)

    means = []

    for j in xrange(11):
        mean = np.nanmean(final[:, j])
        means.append(mean)

    # print means

    reducedlist = []
    for k in xrange(3472):
        has_nan = False
        for j in xrange(len(final[k])):
            if math.isnan(final[k][j]):
                has_nan = True
        if not has_nan:
            reducedlist.append(final[k])

    print "this is reducued list: " + str(len(reducedlist))


    return final


get_data(train)
