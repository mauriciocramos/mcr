from warnings import warn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Reference: https://github.com/datacamp/course-resources-ml-with-experts-budgets/blob/master/src/data/multilabel.py

def multilabel_sample(y, size=1000, min_count=5, seed=None):
    """ Takes a matrix of binary labels `y` and returns
        the indices for a sample of size `size` if
        `size` > 1 or `size` * len(y) if size =< 1.

        The sample is guaranteed to have > `min_count` of
        each label.
    """
    try:
        if (np.unique(y).astype(int) != np.array([0, 1])).any():
            raise ValueError()
    except (TypeError, ValueError):
        raise ValueError('multilabel_sample only works with binary indicator matrices')
    if (y.sum(axis=0) < min_count).any():
        raise ValueError('Some classes do not have enough examples. Change min_count if necessary.')
    if size <= 1:
        size = np.floor(y.shape[0] * size)
    if y.shape[1] * min_count > size:
        msg = "Size less than number of columns * min_count, returning {} items instead of {}."
        warn(msg.format(y.shape[1] * min_count, size))
        size = y.shape[1] * min_count
    rng = np.random.RandomState(seed if seed is not None else np.random.randint(1))  # IF seed is None setS to ZERO?
    if isinstance(y, pd.DataFrame):
        choices = y.index
        y = y.values
    else:
        choices = np.arange(y.shape[0])
    sample_idxs = np.array([], dtype=choices.dtype)
    # first, guarantee > min_count of each label
    for j in range(y.shape[1]):
        label_choices = choices[y[:, j] == 1]
        label_idxs_sampled = rng.choice(label_choices, size=min_count, replace=False)
        sample_idxs = np.concatenate([sample_idxs, label_idxs_sampled])
    sample_idxs = np.unique(sample_idxs)
    # now that we have at least min_count of each, we can just random sample
    sample_count = int(size - sample_idxs.shape[0])
    # get sample_count indices from remaining choices
    remaining_choices = np.setdiff1d(choices, sample_idxs)
    remaining_sampled = rng.choice(remaining_choices,
                                   size=sample_count,
                                   replace=False)
    return rng.permutation(np.concatenate([sample_idxs, remaining_sampled]))


def multilabel_sample_dataframe(df, labels, size, min_count=5, seed=None):
    """ Takes a dataframe `df` and returns a sample of size `size` where all
        classes in the binary matrix `labels` are represented at
        least `min_count` times.
    """
    idxs = multilabel_sample(labels, size=size, min_count=min_count, seed=seed)
    return df.loc[idxs]


def multilabel_train_test_split(X, Y, size, min_count=5, seed=None):
    """ Takes a features matrix `X` and a label matrix `Y` and
        returns (X_train, X_test, Y_train, Y_test) where all
        classes in Y are represented at least `min_count` times.
    """
    index = Y.index if isinstance(Y, pd.DataFrame) else np.arange(Y.shape[0])
    test_set_idxs = multilabel_sample(Y, size=size, min_count=min_count, seed=seed)
    # train_set_idxs = np.setdiff1d(index, test_set_idxs)
    test_set_mask = index.isin(test_set_idxs)
    train_set_mask = ~test_set_mask
    return X[train_set_mask], X[test_set_mask], Y[train_set_mask], Y[test_set_mask]


def sample_report(y, y_sample, figsize=(16, 32)):
    ratio = pd.concat({'sample_ratio': y_sample.mean() / y.mean()}, axis=1)
    ratio.index.name='label__class'
    ratio.sort_values('sample_ratio', ascending=False).plot(kind='barh', stacked=True, figsize=figsize)
    plt.axvline(1, color='red')
    plt.title("Stratified sampling ratios")
    plt.xlabel('ratios')
    plt.show()
    return ratio.sort_values('sample_ratio')


def split_report(y, y_train, y_test, figsize=(16, 32)):
    y_mean = y.mean()
    ratio = pd.concat({'train_ratio': y_train.mean() / y_mean,
                       'test_ratio': y_test.mean() / y_mean}, axis=1)
    ratio.index.name='label__class'
    ratio.sort_values(['test_ratio', 'train_ratio'], ascending=False).plot(kind='barh', stacked=True, figsize=figsize)
    plt.axvline(1, color='red')
    plt.axvline(2, color='red')
    plt.title("Stratified training and testing ratios")
    plt.xlabel('ratios')
    plt.show()
    return ratio.sort_values(['test_ratio', 'train_ratio'])


def sample_split_report(y, y_sample, y_train, y_test, figsize=(16, 32)):
    y_mean = y.mean()
    ratio = pd.concat({'sample_ratio': y_sample.mean() / y_mean,
                       'train_ratio': y_train.mean() / y_mean,
                       'test_ratio': y_test.mean() / y_mean}, axis=1)
    ratio.index.name='label__class'
    ratio.sort_values(['test_ratio',
                       'train_ratio',
                       'sample_ratio'], ascending=False).plot(kind='barh', stacked=True, figsize=figsize)
    plt.axvline(1, color='red')
    plt.axvline(2, color='red')
    plt.axvline(3, color='red')
    plt.title("Stratified sampling, training and splitting ratios")
    plt.xlabel('ratios')
    plt.show()
    return ratio.sort_values(['test_ratio', 'train_ratio', 'sample_ratio'])
