import datetime
import time

import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score

from models.lof import LOF
from models.feature_bagging import FeatureBagging
from models.combination import aom, moa
from utils.stat_models import pearsonr
from utils.utility import get_local_region
from utils.utility import get_competent_detectors
from utils.utility import train_predict_lof, generate_bagging_indices
from utils.utility import print_save_result, save_script
from utils.utility import loaddata, precision_n_score, standardizer

# access the timestamp for logging purpose
today = datetime.datetime.now()
timestamp = today.strftime("%Y%m%d_%H%M%S")

# set numpy parameters
np.set_printoptions(suppress=True, precision=4)

###############################################################################
# parameter settings
# data = 'annthyroid' #
# data = 'arrhythmia'
# data = 'breastw' #
data = 'cardio'
# data = 'glass'
# data = 'ionosphere'
# data = 'letter'
# data = 'lympho'
# data = 'mnist'
# data = 'musk'
# data = 'optdigits'
# data = 'pendigits'
# data = 'pima'
# data = 'satellite'
# data = 'satimage-2'
# data = 'shuttle'
# data = 'speech'
# data = 'thyroid'
# data = 'vertebral'
# data = 'vowels'
# data = 'wbc'

base_detector = 'lof'
n_ite = 20  # number of iterations
test_size = 0.4  # training = 60%, testing = 40%
n_baselines = 11  # the number of baseline algorithms, DO NOT CHANGE

# reference pearson size:
# https://www.researchgate.net/post/What_is_the_minimum_sample_size_to_run_Pearsons_R
loc_region_size = 0
loc_region_min = 30  # min local region size
loc_region_max = 100  # max local region size
###############################################################################
# adjustable parameters
loc_region_perc = 0.1
loc_region_ite = 20  # the number of iterations in defining local region
loc_region_threshold = int(loc_region_ite / 2)  # the threshold to keep a point
loc_min_features = 0.5  # the lower bound of the number of features to use

n_bins = 10
n_selected = 1  # actually not a parameter to tweak

n_clf = 50
k_min = 5
k_max = 200

# for SG_AOM and SG_MOA, choose the right number of buckets
n_buckets = 5
n_clf_bucket = int(n_clf / n_buckets)
assert (n_clf % n_buckets == 0)  # in case wrong number of buckets

# flag for printing and output saving
verbose = True

# record of feature bagging detector
fb_n_neighbors = []
###############################################################################

if __name__ == '__main__':

    start_time = time.time()
    X_orig, y_orig = loaddata(data)

    # initialize the matrix for storing scores
    roc_mat = np.zeros([n_ite, n_baselines])  # receiver operating curve
    ap_mat = np.zeros([n_ite, n_baselines])  # average precision

    for t in range(n_ite):
        print('\nn_ite', t + 1, data)  # print status

        random_state = np.random.RandomState(t)

        # split the data into training and testing
        X_train, X_test, y_train, y_test = train_test_split(X_orig, y_orig,
                                                            test_size=test_size,
                                                            random_state=random_state)
        # in case of small datasets
        if k_max > X_train.shape[0]:
            k_max = X_train.shape[0]
        k_list = random_state.randint(k_min, k_max, size=n_clf).tolist()
        k_list.sort()

        # normalized the data
        X_train_norm, X_test_norm = standardizer(X_train, X_test)

        train_scores = np.zeros([X_train.shape[0], n_clf])
        test_scores = np.zeros([X_test.shape[0], n_clf])

        # initialized the list to store the results
        test_target_list = []
        method_list = []

        # generate a pool of detectors and predict on test instances
        train_scores, test_scores = train_predict_lof(k_list, X_train_norm,
                                                      X_test_norm,
                                                      train_scores,
                                                      test_scores)

        #######################################################################
        # fit feature bagging using median of k_list
        n_neighbors = int(np.median(k_list))
        clf = FeatureBagging(base_estimator=LOF(n_neighbors=n_neighbors),
                             n_estimators=len(k_list))
        print(clf)
        fb_n_neighbors.append(n_neighbors)
        clf.fit(X_train_norm)

        # generate scores
        target_test_feature_bagging = clf.decision_function(X_test_norm)
        test_target_list.append(target_test_feature_bagging)
        method_list.append('FB')
        #######################################################################
        # generate normalized scores
        train_scores_norm, test_scores_norm = standardizer(train_scores,
                                                           test_scores)
        # generate mean and max outputs
        # SG_A and SG_M
        target_test_mean = np.mean(test_scores_norm, axis=1)
        target_test_max = np.max(test_scores_norm, axis=1)
        test_target_list.extend([target_test_mean, target_test_max])
        method_list.extend(['GG_a', 'GG_m'])

        # generate pseudo target for training -> for calculating weights
        target_mean = np.mean(train_scores_norm, axis=1).reshape(-1, 1)
        target_max = np.max(train_scores_norm, axis=1).reshape(-1, 1)

        # generate weighted mean
        # weights are distance or pearson in different modes
        clf_weights_pear = np.zeros([n_clf, 1])
        for i in range(n_clf):
            clf_weights_pear[i] = pearsonr(
                target_mean, train_scores_norm[:, i].reshape(-1, 1))

        # generate weighted mean
        target_test_weighted_pear = np.sum(
            test_scores_norm * clf_weights_pear.reshape(1, -1) /
            clf_weights_pear.sum(), axis=1)

        test_target_list.append(target_test_weighted_pear)
        method_list.append('GG_wa')

        # generate threshold sum
        target_test_threshold = np.sum(test_scores_norm.clip(0), axis=1)
        test_target_list.append(target_test_threshold)
        method_list.append('GG_thresh')

        # generate average of maximum (SG_AOM) and maximum of average (SG_MOA)
        target_test_aom = aom(test_scores_norm, n_buckets, n_clf)
        target_test_moa = moa(test_scores_norm, n_buckets, n_clf)
        test_target_list.extend([target_test_aom, target_test_moa])
        method_list.extend(['GG_aom', 'GG_moa'])
        ##################################################################

        # define the local region size
        loc_region_size = int(X_train_norm.shape[0] * loc_region_perc)
        if loc_region_size < loc_region_min:
            loc_region_size = loc_region_min
        if loc_region_size > loc_region_max:
            loc_region_size = loc_region_max

        # define local region
        ind_arr = get_local_region(X_train_norm, X_test_norm,
                                   loc_region_size,
                                   loc_region_ite=loc_region_ite,
                                   local_region_strength=loc_region_threshold,
                                   loc_min_features=loc_min_features,
                                   random_state=random_state)

        pred_scores_best = np.zeros([X_test.shape[0], ])
        pred_scores_ens = np.zeros([X_test.shape[0], ])

        for i in range(X_test.shape[0]):  # iterate all test instance

            ind_k = ind_arr[i]

            # get the pseudo target: mean
            target_k = target_mean[ind_k,].ravel()

            # get the current scores from all clf
            curr_train_k = train_scores_norm[ind_k, :]

            # initialize containers for correlation
            corr_pear_n = np.zeros([n_clf, ])

            for d in range(n_clf):
                corr_pear_n[d,] = pearsonr(target_k, curr_train_k[:, d])

            # pick the best one
            best_clf_ind = np.nanargmax(corr_pear_n)
            pred_scores_best[i,] = test_scores_norm[i, best_clf_ind]

            pred_scores_ens[i,] = np.max(
                test_scores_norm[
                    i, get_competent_detectors(corr_pear_n, n_bins,
                                               n_selected)])

        test_target_list.extend([pred_scores_best,
                                 pred_scores_ens])
        method_list.extend(['LSCP_a',
                            'LSCP_moa'])
        ######################################################################

        pred_scores_best = np.zeros([X_test.shape[0], ])
        pred_scores_ens = np.zeros([X_test.shape[0], ])

        for i in range(X_test.shape[0]):  # iterate all test instance
            # get the neighbor idx of the current point
            ind_k = ind_arr[i]
            # get the pseudo target: mean
            target_k = target_max[ind_k,].ravel()

            # get the current scores from all clf
            curr_train_k = train_scores_norm[ind_k, :]

            # initialize containers for correlation
            corr_pear_n = np.zeros([n_clf, ])

            for d in range(n_clf):
                corr_pear_n[d,] = pearsonr(target_k, curr_train_k[:, d])

            # pick the best one
            best_clf_ind = np.nanargmax(corr_pear_n)
            pred_scores_best[i,] = test_scores_norm[i, best_clf_ind]

            pred_scores_ens[i,] = np.mean(
                test_scores_norm[
                    i, get_competent_detectors(corr_pear_n, n_bins,
                                               n_selected)])

        test_target_list.extend([pred_scores_best,
                                 pred_scores_ens])
        method_list.extend(['LSCP_m',
                            'LSCP_aom'])

        ######################################################################

        # store performance information and print result
        for i in range(n_baselines):
            roc_mat[t, i] = roc_auc_score(y_test, test_target_list[i])
            ap_mat[t, i] = average_precision_score(y_test,
                                                   test_target_list[i])
            print(method_list[i], roc_mat[t, i])
        print('local region size:', loc_region_size)

    print("--- %s seconds ---" % (time.time() - start_time))
    execution_time = time.time() - start_time

    # save parameters
    save_script(data, base_detector, timestamp, n_ite, test_size, n_baselines,
                loc_region_perc, loc_region_ite, loc_region_threshold,
                loc_min_features, loc_region_size, loc_region_min,
                loc_region_max, n_clf, k_min, k_max, n_bins, n_selected,
                n_buckets, fb_n_neighbors, execution_time)

    # print and save the result
    # default location is /results/***.csv
    print_save_result(data, base_detector, n_baselines, roc_mat,
                      ap_mat, method_list, timestamp, verbose)
