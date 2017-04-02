import csv
import numpy as np
from sklearn import tree


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

    final = final[~(np.isnan(final)).any(1)]

    means = np.mean(final[:, [i for i in xrange(6)]], axis=0)
    stdevs = np.std(final[:, [i for i in xrange(6)]], axis=0)

    # earth = [365.25, 1, 255, 1.3615, 5777, 1]
    # earth = np.asarray(earth)

    # final = np.insert(final, 0, earth, 0)

    return means, stdevs, final


# calculate earth similarity index of planets for labelling
def ESI(data):
    e_flux = 1.3615  # kW/m^2
    e_radius = 6371  # km

    p_radii = data[:, 1] * e_radius
    p_fluxes = data[:, 3] * e_flux

    flux_top = p_fluxes - e_flux
    flux_bottom = p_fluxes + e_flux

    flux = (flux_top / flux_bottom) ** 2

    radii_top = p_radii - e_radius
    radii_bottom = p_radii + e_radius

    radii = (radii_top / radii_bottom) ** 2

    esi = 1 - np.sqrt(0.5 * (flux + radii))

    return esi.reshape((len(esi), 1))


# add classification based on esi
def classify(data, esi):
    # counter = 0
    classification = np.zeros((len(esi), 1))
    for i in xrange(len(data)):
        if esi[i] >= 0.75:
            # counter += 1
            classification[i] = 1
        else:
            classification[i] = -1

    # return (counter / 9200.0) * 1000000000000000000000000
    # return classification
    return np.c_[data, classification]


def train_test(data):
    np.random.shuffle(data)
    train = data[:6900]
    test = data[6900:]
    return train, test


def main():
    file = 'data/cumulative.csv'
    means, stdevs, final = get_data(file)
    esi = ESI(final)

    classed = classify(final, esi)

    train, test = train_test(classed)

    train_points = train[:, :6]
    train_labels = train[:, 6]
    test_points = test[:, :6]
    test_labels = test[:, 6]

    clf = tree.DecisionTreeClassifier()
    clf.fit(train_points, train_labels)

    b = clf.predict(train_points)

    a = clf.predict(test_points)

    Ein = 0.0

    for j in xrange(len(train_points)):
        if b[j] != train_labels[j]:
            Ein += 1

    Eout = 0.0

    for i in xrange(len(test_points)):
        if a[i] != test_labels[i]:
            Eout += 1

    print Eout

main()


