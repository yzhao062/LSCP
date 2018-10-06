import os
import collections
import pathlib

import numpy as np
import scipy.io as scio
from scipy.stats import scoreatpercentile

from sklearn.neighbors import KDTree
from sklearn.metrics import precision_score
from sklearn.preprocessing import StandardScaler
from sklearn.utils import column_or_1d
from sklearn.utils import check_random_state
from sklearn.utils.random import sample_without_replacement
import arff

from models.lof import Lof
from models.knn import Knn


def argmaxp(a, p):
    """Utlity function to return the index of top p values in a
    :param a: list variable
    :param p: number of elements to select
    :return: index of top p elements in a
    """

    a = np.asarray(a).ravel()
    length = a.shape[0]
    pth = np.argpartition(a, length - p)
    return pth[length - p:]


# @njit("i8[:](i8[:], u8, b1)")
def argmaxn(value_list, n, desc=True):
    """
    Return the index of top n elements in the list if order is set to 'desc',
    otherwise return the index of n smallest elements

    :param value_list: a list containing all values
    :type value_list: list, array
    :param n: the number of the elements to select
    :type n: int
    :param order: the order to sort {'desc', 'asc'}
    :type order: str, optional (default='desc')
    :return: the index of the top n elements
    :rtype: list
    """
    value_list = column_or_1d(value_list)
    length = len(value_list)

    # for the smallest n, flip the value
    if not desc:
        n = length - n

    # partition is not part of numba
    value_sorted = np.partition(value_list, length - n)
    threshold = value_sorted[int(length - n)]

    if desc:
        return np.where(np.greater_equal(value_list, threshold))[0]
    else:  # return the index of n smallest elements
        return np.where(np.less(value_list, threshold))[0]


def get_label_n(y, y_pred):
    """ Infer the binary label of the top n samples with highest scores    
    :param y: 
    :param y_pred: 
    :return: 
    """
    out_perc = np.count_nonzero(y) / len(y)
    threshold = scoreatpercentile(y_pred, 100 * (1 - out_perc))
    y_pred = (y_pred > threshold).astype('int')
    return y_pred


def standardizer(X_train, X_test):
    """
    normalization function wrapper
    :param X_train:
    :param X_test:
    :return: X_train and X_test after the Z-score normalization
    """
    scaler = StandardScaler().fit(X_train)
    return scaler.transform(X_train), scaler.transform(X_test)


def precision_n_score(y, y_pred):
    """
    Utlity function to calculate precision@n
    :param y: ground truth
    :param y_pred: number of outliers
    :return: score
    """
    # calculate the percentage of outliers
    out_perc = np.count_nonzero(y) / len(y)

    threshold = scoreatpercentile(y_pred, 100 * (1 - out_perc))
    y_pred = (y_pred > threshold).astype('int')
    return precision_score(y, y_pred)


def loaddata(filename):
    """
    load data
    :param filename:
    :return:
    """
    mat = scio.loadmat(os.path.join('datasets', filename + '.mat'))
    X_orig = mat['X']
    y_orig = mat['y'].ravel()

    return X_orig, y_orig


def read_arff(file_path, misplaced_list):
    misplaced = False
    for item in misplaced_list:
        if item in file_path:
            misplaced = True

    file = arff.load(open(file_path))
    data_value = np.asarray(file['data'])
    attributes = file['attributes']

    X = data_value[:, 0:-2]
    if not misplaced:
        y = data_value[:, -1]
    else:
        y = data_value[:, -2]
    y[y == 'no'] = 0
    y[y == 'yes'] = 1
    y = y.astype('float').astype('int').ravel()

    if y.sum() > len(y):
        print(attributes)
        raise ValueError('wrong sum')

    return X, y


def train_predict_lof(k_list, X_train_norm, X_test_norm, train_scores,
                      test_scores):
    # initialize base detectors
    clf_list = []
    for k in k_list:
        clf = Lof(n_neighbors=k)
        clf.fit(X_train_norm)
        train_score = clf.negative_outlier_factor_ * -1
        test_score = clf.decision_function(X_test_norm) * -1
        clf_name = 'lof_' + str(k)

        clf_list.append(clf_name)
        curr_ind = len(clf_list) - 1

        train_scores[:, curr_ind] = train_score.ravel()
        test_scores[:, curr_ind] = test_score.ravel()

    return train_scores, test_scores


def train_predict_knn(k_list, X_train_norm, X_test_norm, train_scores,
                      test_scores):
    # initialize base detectors
    clf_list = []
    for k in k_list:
        clf = Knn(n_neighbors=k, method='largest')
        clf.fit(X_train_norm)
        train_score = clf.decision_scores
        test_score = clf.decision_function(X_test_norm)
        clf_name = 'knn_' + str(k)

        clf_list.append(clf_name)
        curr_ind = len(clf_list) - 1

        train_scores[:, curr_ind] = train_score.ravel()
        test_scores[:, curr_ind] = test_score.ravel()

    return train_scores, test_scores


def save_script(data, base_detector, timestamp, n_ite, test_size, n_baselines,
                loc_region_perc, loc_region_ite, loc_region_strength,
                loc_min_features, loc_region_size, loc_region_min,
                loc_region_max, n_clf, k_min, k_max, n_bins, n_selected,
                n_buckets, execution_time):
    # initialize the log directory if it does not exist
    pathlib.Path('results').mkdir(parents=True, exist_ok=True)
    f = open(
        'results\\' + data + '_' + base_detector + '_' + timestamp + '.txt',
        'a')

    f.writelines("\n n_ite: " + str(n_ite))
    f.writelines("\n test_size: " + str(test_size))
    f.writelines("\n n_baselines: " + str(n_baselines))
    f.writelines("\n")

    f.writelines("\n loc_region_perc: " + str(loc_region_perc))
    f.writelines("\n loc_region_ite: " + str(loc_region_ite))
    f.writelines("\n loc_region_threshold: " + str(loc_region_strength))
    f.writelines("\n loc_min_features: " + str(loc_min_features))
    f.writelines("\n loc_region_size: " + str(loc_region_size))
    f.writelines("\n loc_region_min: " + str(loc_region_min))
    f.writelines("\n loc_region_max: " + str(loc_region_max))
    f.writelines("\n")

    f.writelines("\n n_clf: " + str(n_clf))
    # f.writelines("\n alpha: " + str(alpha))
    f.writelines("\n k_min: " + str(k_min))
    f.writelines("\n k_max: " + str(k_max))
    f.writelines("\n n_bins: " + str(n_bins))
    f.writelines("\n n_selected: " + str(n_selected))
    f.writelines("\n n_buckets: " + str(n_buckets))

    f.writelines("\n execution_time: " + str(execution_time))
    f.close()


def print_save_result(data, base_detector, n_baselines, n_clf, n_ite, roc_mat,
                      ap_mat, method_list, timestamp, verbose):
    """
    :param data:
    :param base_detector:
    :param n_baselines:
    :param n_clf:
    :param n_ite:
    :param roc_mat:
    :param ap_mat:
    :param prc_mat:
    :param method_list:
    :param timestamp:
    :param verbose:
    :return: None
    """

    roc_scores = np.round(np.mean(roc_mat, axis=0), decimals=4)
    ap_scores = np.round(np.mean(ap_mat, axis=0), decimals=4)

    method_np = np.asarray(method_list)

    top_roc_ind = argmaxp(roc_scores, 1)
    top_ap_ind = argmaxp(ap_scores, 1)

    top_roc_clf = method_np[top_roc_ind].tolist()[0]
    top_ap_clf = method_np[top_ap_ind].tolist()[0]

    top_roc = np.round(roc_scores[top_roc_ind][0], decimals=4)
    top_ap = np.round(ap_scores[top_ap_ind][0], decimals=4)

    roc_diff = np.round(100 * (top_roc - roc_scores) / roc_scores, decimals=4)
    ap_diff = np.round(100 * (top_ap - ap_scores) / ap_scores, decimals=4)

    # initialize the log directory if it does not exist
    pathlib.Path('results').mkdir(parents=True, exist_ok=True)

    # create the file if it does not exist
    f = open(
        'results\\' + data + '_' + base_detector + '_' + timestamp + '.csv',
        'a')

    if verbose:
        f.writelines('method, '
                     'roc, best_roc, diff_roc,'
                     'ap, best_ap, diff_ap,'
                     'best roc, best ap')
    else:
        f.writelines('method, '
                     'roc, ap, p@m,'
                     'best roc, best ap')

    print('method, roc, ap, p@m, best roc, best ap')
    delim = ','
    for i in range(n_baselines):
        print(method_list[i], roc_scores[i], ap_scores[i],
              top_roc_clf, top_ap_clf)

        if verbose:
            f.writelines(
                '\n' + str(method_list[i]) + delim +
                str(roc_scores[i]) + delim + str(top_roc) + delim + str(
                    roc_diff[i]) + delim +
                str(ap_scores[i]) + delim + str(top_ap) + delim + str(
                    ap_diff[i]) + delim +
                top_roc_clf + delim + top_ap_clf)
        else:
            f.writelines(
                '\n' + str(method_list[i]) + delim +
                str(roc_scores[i]) + delim +
                str(ap_scores[i]) + delim +
                top_roc_clf + delim + top_ap_clf)

    f.close()


def generate_bagging_indices(random_state, bootstrap_features, n_features,
                             min_features, max_features):
    """
    Randomly draw feature indices. Internal use only.

    Modified from sklearn/ensemble/bagging.py
    """
    # Get valid random state
    random_state = check_random_state(random_state)

    # decide number of features to draw
    random_n_features = random_state.randint(min_features, max_features)

    # Draw indices
    feature_indices = _generate_indices(random_state, bootstrap_features,
                                        n_features, random_n_features)

    return feature_indices


def _generate_indices(random_state, bootstrap, n_population, n_samples):
    """
    Draw randomly sampled indices. Internal use only.

    See sklearn/ensemble/bagging.py
    """
    # Draw sample indices
    if bootstrap:
        indices = random_state.randint(0, n_population, n_samples)
    else:
        indices = sample_without_replacement(n_population, n_samples,
                                             random_state=random_state)

    return indices


def get_local_region(X_train_norm, X_test_norm, loc_region_size,
                     loc_region_ite, local_region_strength,
                     loc_min_features, random_state):
    # Initialize the local region list
    grid = [[]] * X_test_norm.shape[0]

    for t in range(loc_region_ite):
        features = generate_bagging_indices(random_state,
                                            bootstrap_features=False,
                                            n_features=X_train_norm.shape[1],
                                            min_features=int(
                                                X_train_norm.shape[
                                                    1] * loc_min_features),
                                            max_features=X_train_norm.shape[1])

        tree = KDTree(X_train_norm[:, features])
        dist_arr, ind_arr = tree.query(X_test_norm[:, features],
                                       k=loc_region_size)

        for j in range(X_test_norm.shape[0]):
            grid[j] = grid[j] + ind_arr[j, :].tolist()

    grid_f = [[]] * X_test_norm.shape[0]
    for j in range(X_test_norm.shape[0]):
        grid_f[j] = [item for item, count in
                     collections.Counter(grid[j]).items() if
                     count > local_region_strength]

    return grid_f


def get_competent_detectors(scores, n_bins=10, n_selected=5):
    """ algorithm for selecting the most competent detectors
    :param scores:
    :param n_bins:
    :param n_selected:
    :return:
    """
    scores = scores.reshape(-1, 1)
    hist, bin_edges = np.histogram(scores, bins=n_bins)
    #    dense_bin = np.argmax(hist)
    max_bins = argmaxn(hist, n=n_selected, desc=True)
    candidates = []
    #    print(hist)
    for max_bin in max_bins:
        #        print(bin_edges[max_bin], bin_edges[max_bin+1])
        selected = np.where((scores >= bin_edges[max_bin])
                            & (scores <= bin_edges[max_bin + 1]))
        #        print(selected)
        candidates = candidates + selected[0].tolist()

    #    print(np.mean(scores[candidates,:]), np.mean(scores))
    # return np.mean(scores[candidates, :])
    return candidates
