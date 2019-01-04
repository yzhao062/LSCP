import numpy as np
from sklearn.utils.validation import check_array
from sklearn.utils.validation import column_or_1d
from sklearn.utils.testing import assert_equal


def aom(scores, n_buckets, n_estimators, standard=True):
    '''
    Average of Maximum - An ensemble method for outlier detection

    Aggarwal, C.C. and Sathe, S., 2015. Theoretical foundations and algorithms
    for outlier ensembles. ACM SIGKDD Explorations Newsletter, 17(1), pp.24-47.

    :param scores:
    :param n_buckets:
    :param n_estimators:
    :param standard:
    :return:
    '''
    scores = np.asarray(scores)
    if scores.shape[1] != n_estimators:
        raise ValueError('score matrix should be n_samples by n_estimaters')

    scores_aom = np.zeros([scores.shape[0], n_buckets])

    n_estimators_per_bucket = int(n_estimators / n_buckets)
    if n_estimators % n_buckets != 0:
        Warning('n_estimators / n_buckets leads to a remainder')

    # shuffle the estimator order
    estimators_list = list(range(0, n_estimators, 1))
    np.random.shuffle(estimators_list)

    head = 0
    for i in range(0, n_estimators, n_estimators_per_bucket):
        tail = i + n_estimators_per_bucket
        batch_ind = int(i / n_estimators_per_bucket)

        scores_aom[:, batch_ind] = np.max(
            scores[:, estimators_list[head:tail]], axis=1)

        head = head + n_estimators_per_bucket

    return np.mean(scores_aom, axis=1)


def moa(scores, n_buckets, n_estimators):
    '''
    Maximum of Average - An ensemble method for outlier detection

    Aggarwal, C.C. and Sathe, S., 2015. Theoretical foundations and algorithms
    for outlier ensembles. ACM SIGKDD Explorations Newsletter, 17(1), pp.24-47.

    :param scores:
    :param n_buckets:
    :param n_estimators:
    :param standard:
    :return:
    '''
    scores = np.asarray(scores)
    if scores.shape[1] != n_estimators:
        raise ValueError('score matrix should be n_samples by n_estimaters')

    scores_moa = np.zeros([scores.shape[0], n_buckets])

    n_estimators_per_bucket = int(n_estimators / n_buckets)
    if n_estimators % n_buckets != 0:
        Warning('n_estimators / n_buckets leads to a remainder')

    # shuffle the estimator order
    estimators_list = list(range(0, n_estimators, 1))
    np.random.shuffle(estimators_list)

    head = 0
    for i in range(0, n_estimators, n_estimators_per_bucket):
        tail = i + n_estimators_per_bucket
        batch_ind = int(i / n_estimators_per_bucket)

        scores_moa[:, batch_ind] = np.mean(
            scores[:, estimators_list[head:tail]], axis=1)

        head = head + n_estimators_per_bucket

    return np.max(scores_moa, axis=1)


def average(scores, estimator_weight=None):
    """Combination method to merge the outlier scores from multiple estimators
    by taking the average.

    Parameters
    ----------
    scores : numpy array of shape (n_samples, n_estimators)
        Score matrix from multiple estimators on the same samples.

    estimator_weight : list of shape (1, n_estimators)
        If specified, using weighted average

    Returns
    -------
    combined_scores : numpy array of shape (n_samples, )
        The combined outlier scores.

    """
    scores = check_array(scores)

    if estimator_weight is not None:
        estimator_weight = column_or_1d(estimator_weight).reshape(1, -1)
        assert_equal(scores.shape[1], estimator_weight.shape[1])

        # (d1*w1 + d2*w2 + ...+ dn*wn)/(w1+w2+...+wn)
        # generated weighted scores
        scores = np.sum(np.multiply(scores, estimator_weight),
                        axis=1) / np.sum(
            estimator_weight)
        return scores.ravel()

    else:
        return np.mean(scores, axis=1).ravel()


def maximization(scores):
    """Combination method to merge the outlier scores from multiple estimators
    by taking the maximum.

    Parameters
    ----------
    scores : numpy array of shape (n_samples, n_estimators)
        Score matrix from multiple estimators on the same samples.

    Returns
    -------
    combined_scores : numpy array of shape (n_samples, )
        The combined outlier scores.

    """

    scores = check_array(scores)
    return np.max(scores, axis=1).ravel()
