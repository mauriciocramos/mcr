from sklearn.model_selection import validation_curve
import matplotlib.pyplot as plt
import numpy as np
from time import time
from datetime import datetime


def plot_validation_curve(estimator, X, y, param_name, param_range, param_label,
                          cv=None, scoring=None, n_jobs=None, verbose=0, xscale='linear'):
    print('Started', datetime.now().isoformat(timespec='minutes'))
    t = time()
    train_scores, test_scores = validation_curve(estimator, X, y, param_name=param_name, param_range=param_range,
                                                 cv=cv, scoring=scoring, n_jobs=n_jobs, verbose=verbose)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()
    plt.title(param_label + ' validation curve')
    plt.xlabel(param_label + ' (log scale)' if xscale=='log' else '')
    plt.xscale(xscale)
    plt.ylabel("Score")
    #plt.ylim(0.0, 1.1)
    param_range = [repr(x) if isinstance(x, tuple) else x for x in param_range]
    plt.plot(param_range, train_scores_mean, 'o-', label="Training", color="r")
    plt.fill_between(param_range, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.2, color="r")
    plt.plot(param_range, test_scores_mean, 'o-', label="Validation", color="g")
    plt.fill_between(param_range, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.2, color="g")
    plt.text(param_range[0], #train_sizes.min(),
             test_scores_mean.max(),
             '{:.4f}'.format(test_scores_mean.max()),
             va='bottom', ha='left', size=10, color='g').set_bbox(dict(facecolor='white', alpha=1, edgecolor='green'))
    plt.suptitle('Elapsed {:.1f} minutes'.format((time() - t) / 60), x=0, y=0, va='bottom', ha='left')
    plt.legend(loc="best")
    return plt
