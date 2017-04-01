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


def main():
    train = 'data/cumulative.csv'
    means, final = get_data(train)



main()
