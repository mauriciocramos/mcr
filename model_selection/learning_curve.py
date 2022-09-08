from sklearn.model_selection import learning_curve
import matplotlib.pyplot as plt
import numpy as np
import warnings
from time import time


def plot_learning_curve(estimator, title, X, y, axes=None, ylim=None, cv=None, n_jobs=None, scoring=None,
                        train_sizes=np.linspace(.1, 1.0, 5), verbose=0):
    """
    Ref: https://scikit-learn.org/stable/auto_examples/model_selection/plot_learning_curve.html

    Generate 5 plots: the test and training learning curve, the training
    samples vs fit times curve, the fit times vs test score curve,
    the training samples vs test score times curve, the score times vs test score curve

    Parameters
    ----------
    estimator : object type that implements the "fit" and "predict" methods
        An object of that type which is cloned for each validation.
    
    title : str
    Title for the chart.

    X : array-like, shape (n_samples, n_features)
        Training vector, where n_samples is the number of samples and
        n_features is the number of features.

    y : array-like, shape (n_samples) or (n_samples, n_features), optional
        Target relative to X for classification or regression;
        None for unsupervised learning.

    axes : array-like of shape (5,), default=None
        Axes to use for plotting the curves.

    ylim : tuple of shape (2,), default=None
        Defines minimum and maximum y-values plotted, e.g. (ymin, ymax).

    cv : int, cross-validation generator or an iterable, default=None
        Determines the cross-validation splitting strategy.
        Possible inputs for cv are:

          - None, to use the default 5-fold cross-validation,
          - integer, to specify the number of folds.
          - :term:`CV splitter`,
          - An iterable yielding (train, test) splits as arrays of indices.

        For integer/None inputs, if ``y`` is binary or multiclass,
        :class:`StratifiedKFold` used. If the estimator is not a classifier
        or if ``y`` is neither binary nor multiclass, :class:`KFold` is used.

        Refer :ref:`User Guide <cross_validation>` for the various
        cross-validators that can be used here.

    n_jobs : int or None, default=None
        Number of jobs to run in parallel.
        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
        for more details.

    scoring : str or callable, default=None
        A str (see model evaluation documentation) or
        a scorer callable object / function with signature
        ``scorer(estimator, X, y)``.

    train_sizes : array-like of shape (n_ticks,)
        Relative or absolute numbers of training examples that will be used to
        generate the learning curve. If the ``dtype`` is float, it is regarded
        as a fraction of the maximum size of the training set (that is
        determined by the selected validation method), i.e. it has to be within
        (0, 1]. Otherwise it is interpreted as absolute sizes of the training
        sets. Note that for classification the number of samples usually have
        to be big enough to contain at least one sample from each class.
        (default: np.linspace(0.1, 1.0, 5))
    """
    t = time()
    if axes is None:
        # TODO: try constrainted_layout=False
        fig, axes = plt.subplots(nrows=1, ncols=5, figsize=(16, 16/5), constrained_layout=True)

    axes[0].set_title('Learning curve')
    if ylim is not None:
        axes[0].set_ylim(*ylim)
    axes[0].set_xlabel("Training examples")
    axes[0].set_ylabel("Scores")

    train_sizes, train_scores, test_scores, fit_times, score_times = \
        learning_curve(estimator, X, y, train_sizes=train_sizes, cv=cv, scoring=scoring, n_jobs=n_jobs,
                       verbose=verbose, return_times=True)

    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    fit_times_mean = np.mean(fit_times, axis=1)
    fit_times_std = np.std(fit_times, axis=1)
    score_times_mean = np.mean(score_times, axis=1)
    score_times_std = np.std(score_times, axis=1)

    # Plot learning curve
    axes[0].grid()
    axes[0].fill_between(train_sizes, train_scores_mean - train_scores_std,
                         train_scores_mean + train_scores_std, alpha=0.1,
                         color="r")
    axes[0].fill_between(train_sizes, test_scores_mean - test_scores_std,
                         test_scores_mean + test_scores_std, alpha=0.1,
                         color="g")
    axes[0].plot(train_sizes, train_scores_mean, 'o-', color="r",
                 # label="Training score")
                 label="Training")
    axes[0].plot(train_sizes, test_scores_mean, 'o-', color="g",
                 # label="Cross-validation score")
                 label="Cross-validation")
    # annotate the highest test score mean
    axes[0].text(train_sizes.min(),
                 test_scores_mean.max(),
                 '{:.4f}'.format(test_scores_mean.max()),
                 va='center', ha='left', size=10, color='g').set_bbox(dict(facecolor='white', alpha=1, edgecolor='green'))
    axes[0].legend(loc="best")

    # Plot n_samples vs fit_times
    axes[1].grid()
    axes[1].plot(train_sizes, fit_times_mean, 'o-', color='r')
    axes[1].fill_between(train_sizes, fit_times_mean - fit_times_std,
                         fit_times_mean + fit_times_std, alpha=0.1)
    axes[1].set_xlabel("Training examples")
    axes[1].set_ylabel("Fit time (s)")
    axes[1].set_title("Training scalability")

    # Plot fit_time vs score
    # new: sort fit times
    fit_time_argsort = fit_times_mean.argsort()
    fit_time_sorted = fit_times_mean[fit_time_argsort]
    test_scores_mean_sorted = test_scores_mean[fit_time_argsort]
    test_scores_std_sorted = test_scores_std[fit_time_argsort]
    #
    axes[2].grid()
    # axes[2].plot(fit_times_mean, test_scores_mean, 'o-', color='g')
    axes[2].plot(fit_time_sorted, test_scores_mean_sorted, 'o-', color='g')
    
    # axes[2].fill_between(fit_times_mean, test_scores_mean - test_scores_std,
                         # test_scores_mean + test_scores_std, alpha=0.1)
    axes[2].fill_between(fit_time_sorted, test_scores_mean_sorted - test_scores_std_sorted,
                         test_scores_mean_sorted + test_scores_std_sorted, alpha=0.1)
    
    axes[2].set_xlabel("Fit time (s)")
    axes[2].set_ylabel("Cross-validation score")
    axes[2].set_title("Model performance")

    # Plot n_samples vs score_times
    axes[3].grid()
    axes[3].plot(train_sizes, score_times_mean, 'o-', color='g')
    axes[3].fill_between(train_sizes, score_times_mean - score_times_std,
                         score_times_mean + score_times_std, alpha=0.1)
    axes[3].set_xlabel("Training examples")
    axes[3].set_ylabel("Cross-validation score time (s)")
    axes[3].set_title("Cross-validation scalability")

    # Plot score_time vs score
    # new: sort fit times
    score_time_argsort = score_times_mean.argsort()
    score_time_sorted = score_times_mean[score_time_argsort]
    test_scores_mean_sorted = test_scores_mean[score_time_argsort]
    test_scores_std_sorted = test_scores_std[score_time_argsort]
    #
    axes[4].grid()
    # axes[4].plot(score_times_mean, test_scores_mean, 'o-', color='g')
    axes[4].plot(score_times_sorted, test_scores_mean_sorted, 'o-', color='g')
    # axes[4].fill_between(score_times_mean, test_scores_mean - test_scores_std,
    #                      test_scores_mean + test_scores_std, alpha=0.1)
    axes[4].fill_between(score_time_sorted, test_scores_mean_sorted - test_scores_std_sorted,
                         test_scores_mean_sorted + test_scores_std_sorted, alpha=0.1)
    axes[4].set_xlabel("Cross-validation score time (s)")
    axes[4].set_ylabel("Cross-validation score")
    axes[4].set_title("Cross-validation performance")

    fig.suptitle('{} in {:.1f} minutes'.format(title, np.floor(time() - t) / 60))

    return plt
