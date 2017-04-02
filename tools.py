import csv
import numpy as np
import math
import random


def get_data(filename):
    with open(filename, 'r') as data_file:
        reader = csv.reader(data_file, delimiter=',', quotechar='"')
        raw_data = [row for row in reader]
    raw_data = raw_data[54:]
    data_array = np.asarray(raw_data, dtype=str)

    columns = [11, 26, 29, 32, 38, 44]
    final = data_array[:, columns]

    t1 = np.nan
    t2 = str(t1)
    final[final == ''] = t2

    final = final.astype(float)
    final = final[~(np.isnan(final)).any(1)]

    return final


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


def classify(data, esi):
    classification = np.zeros((len(esi), 1))
    for i in xrange(len(data)):
        if esi[i] >= 0.85:
            classification[i] = 1
        else:
            classification[i] = -1
    return np.c_[data, classification]


def train_test(data):
    np.random.shuffle(data)
    train = data[:6900]
    test = data[6900:]
    return train, test