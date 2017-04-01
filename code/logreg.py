import csv
import numpy as np
import math
import random
import matplotlib.pyplot as plt

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


# histogram plotting function to determin threshold esi
# def plot_hist(title, xlabel, ylabel, data):
#     plt.title(title)
#     plt.xlabel(xlabel)
#     plt.ylabel(ylabel)
#     plt.hist(data)
#     plt.show()
#     return 1


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


# Split data up into training and testing data
def train_test(data):
    np.random.shuffle(data)
    train = data[:6900]
    test = data[6900:]
    return train, test


'''
normalize training data by the rule feature - mean of
the column the feature is in, divided by the standard
deviation of the column the feature is in.
'''
def normalize_train(data):
    # bias term
    normalized = np.ones((len(data), 1))
    means = np.mean(data[:, [i for i in xrange(6)]], axis=0)
    stdevs = np.std(data[:, [j for j in xrange(6)]], axis=0)

    for i in xrange(len(data[0])):
        i_col = data[:, i]
        i_col = (i_col - means[i]) / (stdevs[i])
        normalized = np.c_[normalized, i_col]

    return normalized[:,1:7], means, stdevs


'''
normalize testing data via the same rule before
but wrt to the training data means and stdevs
'''
def normalize_test(points, means, stdevs):
    normalized = np.ones((len(points), 1))

    for i in xrange(len(points[0])):
        i_column = points[:, i]
        i_column = (i_column - means[i - 1]) / stdevs[i - 1]
        normalized = np.c_[normalized, i_column]

    # return normalized[:,1:len(points[0])+1]
    return normalized[:,1:7]


'''
Function that calculates l2-loss of a set of data points
given a set of model weights. Returns the total loss
'''
def l2_loss(data, labels, weights):
    tot_loss = 0

    for i in xrange(len(data)):
        exponent = -1.0 * labels[i] * np.dot(weights, data[i])
        tot_loss += math.log(1 + np.exp(exponent))

    return tot_loss


'''
Calculates gradient of the l2-logistic loss function
Returns the gradient of the function given a single point,
its respective classification, a set of weights, a lambda,
and the number of the points in the dataset.
'''
def gradient_log_reg(point, label, weights, lamb, N):
    left = (2.0 * lamb * weights) / N
    exponent = label * np.dot(weights, point)
    right_top = -1.0 * label * point
    right_bottom = np.exp(exponent) + 1.0
    right = right_top / right_bottom

    return left + right


'''
Weight update function defined as the difference
between the previous weights and the product
of the step value and the gradient of the loss at a point
'''
def weight_update(points, labels, step, weights, lamb, N):
    i = random.randint(0, N - 1)
    new_weight = weights - step * gradient_log_reg(points[i], labels[i], weights, lamb, N)

    return new_weight


'''
Performs stochastic gradient descent until the quotient of
the loss reduction between two epochs and the
initial loss reduction is less than a defined epsilon.
'''
def sgd(points, labels, step, weights, lamb, N):
    original_loss = l2_loss(points, labels, weights)
    epsilon = 0.0001
    first_loss = 0
    first_loss_reduction = 0
    contin = True

    for i in xrange(1000):
        weights = weight_update(points, labels, step, weights, lamb, N)

    first_loss = l2_loss(points, labels, weights)
    first_loss_reduction = original_loss - first_loss

    print "epsilon: " + str(epsilon)

    while contin:
        prev_error = l2_loss(points, labels, weights)

        for j in xrange(1000):
            weights = weight_update(points, labels, step, weights, lamb, N)

        cur_error = l2_loss(points, labels, weights)

        cur_loss_reduction = prev_error - cur_error
        print "cur_loss_reduction / first_loss_reduction: " + str(cur_loss_reduction / first_loss_reduction)

        if (cur_loss_reduction / first_loss_reduction) < epsilon:
            contin = False

        final_loss = l2_loss(points, labels, weights)

        return weights, final_loss


'''
Plotting function for Ein and Eout of models
'''
def plot_it(x_coord, y_coord, title, xlabel, ylabel):
    plt.plot(x_coord, y_coord)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.show()


def main():
    file = 'data/cumulative.csv'
    means, stdevs, final = get_data(file)
    esi = ESI(final)

    # plot_hist("ESI Distribution", "ESI", "Frequency", esi)
    '''
    Plotted ESI distribution, chose 0.75 ESI score as threshold
    for classification, anything about 0.75 is +1 and earthlike
    and anything under 0.75 is -1 and not earthlike
    '''
    classed = classify(final, esi)

    train, test = train_test(classed)

    train_points = train[:, :6]
    train_labels = train[:, 6]
    test_points = test[:, :6]
    test_labels = test[:, 6]

    norm_train_points, means, stdevs = normalize_train(train_points)
    norm_test_points = normalize_test(test_points, means, stdevs)

    # print norm_test_points

    train_len = float(len(train_points))
    test_len = float(len(test_points))

    train_error = []
    test_error = []

    weights = np.random.uniform(-10**-5, 10**-5, size=6)
    # lambdas = [(0.0001 * (5.0**i)) for i in xrange(15)]

    # for k in xrange(15):
    #     final_weights, final_loss = sgd(norm_train_points, train_labels, (10 ** -5), weights, lambdas[k], train_len)
    #     train_error.append(final_loss / train_len)
    #     Eout = l2_loss(norm_test_points, test_labels, final_weights)
    #     test_error.append(Eout / test_len)

    lambd = 0.005
    final_weights, final_loss = sgd(norm_train_points, train_labels, (10 ** -15), weights, lambd, train_len)
    print final_loss / train_len

    # plot_it(lambdas, train_error, "Ein vs Lambdas", "Lambdas", "Ein")
    # plot_it(lambdas, test_error, "Eout vs Lambdas", "Lambdas", "Eout")



    return 0


main()
