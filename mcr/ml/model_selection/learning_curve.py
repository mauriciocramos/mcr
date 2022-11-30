from sklearn import clone
from sklearn.model_selection import learning_curve
import matplotlib.pyplot as plt
import numpy as np
from time import time


def plot_learning_curve(estimator, X, y, title=None, fig=None, axes=None, ylim=None, cv=None, n_jobs=None, scoring=None,
                        train_sizes=np.linspace(.2, 1.0, 5), verbose=0, alpha=0.2, figsize=(19.2, 19.2/5)):
    """
    Ref: https://scikit-learn.org/stable/auto_examples/model_selection/plot_learning_curve.html

    Generate 5 plots: the test and training learning curve, the training
    samples vs fit times curve, the fit times vs test score curve,
    the training samples vs test score times curve, the score times vs test score curve

    Parameters
    ----------
    :param estimator: object type that implements the "fit" and "predict" methods
        An object of that type which is cloned for each validation.
    :param title: str
    Title for the chart.
    :param X: array-like, shape (n_samples, n_features)
        Training vector, where n_samples is the number of samples and
        n_features is the number of features.
    :param y: array-like, shape (n_samples) or (n_samples, n_features), optional
        Target relative to X for classification or regression;
        None for unsupervised learning.
    :param fig: matplotlib.figure.Figure
        matplotlib.figure.Figure under which axes are plotted
    :param axes: array-like of shape (5,), default=None
        Axes to use for plotting the curves.
    :param ylim: tuple of shape (2,), default=None
        Defines minimum and maximum y-values plotted, e.g. (ymin, ymax).
    :param cv: int, cross-validation generator or an iterable, default=None
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
    :param n_jobs: int or None, default=None
        Number of jobs to run in parallel for training sets.
        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
        for more details.
    :param scoring: str or callable, default=None
        A str (see model evaluation documentation) or
        a scorer callable object / function with signature
        ``scorer(estimator, X, y)``.
    :param train_sizes: array-like of shape (n_ticks,)
        Relative or absolute numbers of training examples that will be used to
        generate the learning curve. If the ``dtype`` is float, it is regarded
        as a fraction of the maximum size of the training set (that is
        determined by the selected validation method), i.e. it has to be within
        (0, 1]. Otherwise it is interpreted as absolute sizes of the training
        sets. Note that for classification the number of samples usually have
        to be big enough to contain at least one sample from each class.
        (default: np.linspace(0.2, 1.0, 5))
    :param verbose: int, default=0
            The verbosity level: if non zero, progress messages are
            printed. Above 50, the output is sent to stdout.
            The frequency of the messages increases with the verbosity level.
            If it more than 10, all iterations are reported.
    :param alpha: float, from 0 to 1, default=0.1
            The alpha blending value, between 0 (transparent) and 1 (opaque) for the error bands.
    :param figsize:
    :return: None
    """

    t = time()
    if axes is None:
        fig, axes = plt.subplots(nrows=1, ncols=5, figsize=figsize, constrained_layout=True)
    fig.suptitle(title)
    # '{} in {:.1f} minutes'.format(title, np.floor(time() - t) / 60)

    # region Determines cross-validated training and test scores for different training set sizes.
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
    # endregion
    
    # region Plot learning curve
    axes[0].set_title('Learning curve')
    if ylim is not None:
        axes[0].set_ylim(*ylim)
    axes[0].set_xlabel("Training examples")
    axes[0].set_ylabel("Scores")
    axes[0].grid(color='gray', linestyle=':')
    axes[0].fill_between(train_sizes, train_scores_mean - train_scores_std,
                         train_scores_mean + train_scores_std, alpha=alpha,
                         color="r")
    axes[0].fill_between(train_sizes, test_scores_mean - test_scores_std,
                         test_scores_mean + test_scores_std, alpha=alpha,
                         color="g")
    axes[0].plot(train_sizes, train_scores_mean, 'o-', color="r",
                 label="Train")
    axes[0].plot(train_sizes, test_scores_mean, 'o-', color="g",
                 label="CV")
    # annotate the highest score mean
    axes[0].text(train_sizes[train_scores_mean.argsort()[-1]], train_scores_mean.max(),
                 '{:.4f}'.format(train_scores_mean.max()),
                 va='bottom', ha='right', size=10, color='r').set_bbox(dict(facecolor='black', alpha=.5, edgecolor='red'))
    axes[0].text(train_sizes[test_scores_mean.argsort()[-1]], test_scores_mean.max(),
                 '{:.4f}'.format(test_scores_mean.max()),
                 va='top', ha='right', size=10, color='g').set_bbox(dict(facecolor='black', alpha=.5, edgecolor='green'))
    axes[0].legend(loc="best")
    # endregion

    # region Plot n_samples vs fit_times
    axes[1].grid(color='gray', linestyle=':')
    axes[1].plot(train_sizes, fit_times_mean, 'o-', color='r')
    axes[1].fill_between(train_sizes, fit_times_mean - fit_times_std,
                         fit_times_mean + fit_times_std, alpha=alpha, color='r')
    axes[1].set_xlabel("Training examples")
    axes[1].set_ylabel("Fit time (s)")
    axes[1].set_title("Training scalability")
    # endregion

    # region Plot fit_time vs score
    # new: sort fit times
    # fit_time_argsort = fit_times_mean.argsort()
    # fit_time_sorted = fit_times_mean[fit_time_argsort]
    # test_scores_mean_sorted = test_scores_mean[fit_time_argsort]
    # test_scores_std_sorted = test_scores_std[fit_time_argsort]
    #
    axes[2].grid(color='gray', linestyle=':')
    axes[2].plot(fit_times_mean, test_scores_mean, 'o-', color='g')
    # axes[2].plot(fit_time_sorted, test_scores_mean_sorted, 'o-', color='g')
    
    axes[2].fill_between(fit_times_mean, test_scores_mean - test_scores_std,
                         test_scores_mean + test_scores_std, alpha=alpha, color='g')
    # axes[2].fill_between(fit_time_sorted, test_scores_mean_sorted - test_scores_std_sorted,
    #                      test_scores_mean_sorted + test_scores_std_sorted, alpha=alpha, color='g')
    
    axes[2].set_xlabel("Fit time (s)")
    axes[2].set_ylabel("Cross-validation score")
    axes[2].set_title("Model performance")
    # endregion

    # region Plot n_samples vs score_times
    axes[3].grid(color='gray', linestyle=':')
    axes[3].plot(train_sizes, score_times_mean, 'o-', color='g')
    axes[3].fill_between(train_sizes, score_times_mean - score_times_std,
                         score_times_mean + score_times_std, alpha=alpha, color='g')
    axes[3].set_xlabel("Training examples")
    axes[3].set_ylabel("Cross-validation score time (s)")
    axes[3].set_title("Cross-validation scalability")
    # endregion

    # region Plot score_time vs score
    # new: sort score times
    # score_time_argsort = score_times_mean.argsort()
    # score_time_sorted = score_times_mean[score_time_argsort]
    # test_scores_mean_sorted = test_scores_mean[score_time_argsort]
    # test_scores_std_sorted = test_scores_std[score_time_argsort]
    #
    axes[4].grid(color='gray', linestyle=':')
    axes[4].plot(score_times_mean, test_scores_mean, 'o-', color='g')
    # axes[4].plot(score_time_sorted, test_scores_mean_sorted, 'o-', color='g')
    axes[4].fill_between(score_times_mean, test_scores_mean - test_scores_std,
                         test_scores_mean + test_scores_std, alpha=alpha, color='g')
    # axes[4].fill_between(score_time_sorted, test_scores_mean_sorted - test_scores_std_sorted,
    #                      test_scores_mean_sorted + test_scores_std_sorted, alpha=alpha, color='g')
    axes[4].set_xlabel("Cross-validation score time (s)")
    axes[4].set_ylabel("Cross-validation score")
    axes[4].set_title("Cross-validation performance")
    # endregion

    fig.supxlabel(f'{(time() - t) / 60:.2f} min', x=0, y=0, va='bottom', ha='left')


def plot_multi_learning_curves(estimator, X, y, parameter_grid, fig=None, axes=None, ylim=None, cv=None, n_jobs=None,
                               scoring=None, train_sizes=np.linspace(0.2, 1, 5), verbose=0, alpha=0.2,
                               figsize=(19.2, 19.2/5)):
    # TODO: let plot return scoring and ploti_multi returns the best trained model
    for parameter in parameter_grid:
        cloned_pl = clone(estimator)
        cloned_pl.set_params(**parameter)
        title = ', '.join([k.split('__')[-1]+'='+str(' '.join(repr(v).split())) for k, v in parameter.items()])
        title = title[0:len(title)//2] + '\n' + title[len(title)//2:]
        plot_learning_curve(estimator=cloned_pl,
                            X=X,
                            y=y,
                            title=title,
                            fig=fig,
                            axes=axes,
                            ylim=ylim,
                            cv=cv,
                            n_jobs=n_jobs,
                            scoring=scoring,
                            train_sizes=train_sizes,
                            verbose=verbose,
                            alpha=alpha,
                            figsize=figsize)
        plt.show()
