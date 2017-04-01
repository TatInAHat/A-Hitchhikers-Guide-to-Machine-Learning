import csv
import numpy as np
import math

train = 'data/cumulative.csv'

# preprocess data


def get_data(filename):
    # extract data from cumulative.csv file
    with open(filename, 'r') as data_file:
        reader = csv.reader(data_file, delimiter=',', quotechar='"')
        raw_data = [row for row in reader]
    # take only row 54 onwards, where the actual data is present
    raw_data = raw_data[54:]
    data_array = np.asarray(raw_data, dtype=str)

    # columns we want from data
    columns = [11, 26, 29, 32, 38, 44]
    # get only columns from data
    final = data_array[:, columns]

    # replace all empty strings with nans
    t1 = np.nan
    t2 = str(t1)
    final[final == ''] = t2

    final = final.astype(float)

    means = np.nanmean(final[:, [i for i in xrange(6)]], axis=0)

    final = final[~(np.isnan(final)).any(1)]

    return means, final


def main():
    train = 'data/cumulative.csv'
    means, final = get_data(train)


main()
