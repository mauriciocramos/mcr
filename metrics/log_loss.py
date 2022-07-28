import numpy as np
import pandas as pd
from IPython.display import display
from sklearn.metrics import classification_report
from time import time
from datetime import datetime


def log_loss(actual, predicted, eps=1e-15):  # sklearn eps: 1e-15, sys.float_info: 2.22e-16. Datacamp: eps=1e-14
    """ Computes the logarithmic loss between predicted and actual when these are 1D arrays
        Reference: https://www.datacamp.com/courses/machine-learning-with-the-experts-school-budgets
        :param predicted: The predicted probabilities as floats between 0-1
        :param actual: The actual binary labels. Either 0 or 1.
        :param eps: (optional) log(0) is inf, so we need to offset our predicted values slightly by eps from 0 or 1.
    """
    predicted = np.clip(predicted, eps, 1 - eps)
    loss = -1 * np.mean(actual * np.log(predicted)
                        + (1 - actual)
                        * np.log(1 - predicted))
    return loss


def multi_multi_log_loss(actual, predicted, class_column_indices=None, averaged=True, eps=1e-15):
    """ Multi class version of Logarithmic Loss metric as implemented on
        DrivenData.org
        Reference: https://github.com/datacamp/course-resources-ml-with-experts-budgets/blob/master/src/models/metrics.py
    """

    # Default flatten indices
    if class_column_indices is None:
        class_column_indices = [[i] for i in range(actual.shape[1])]
    class_scores = np.ones(len(class_column_indices), dtype=np.float64)
    # avoids add .values in actual upstream
    if isinstance(actual, pd.DataFrame):
        actual = actual.values
    # calculate log loss for each set of columns that belong to a class:
    for k, this_class_indices in enumerate(class_column_indices):
        # get just the columns for this class
        preds_k = predicted[:, this_class_indices].astype(np.float64)
        # normalize so probabilities sum to one (unless sum is zero, then we clip)
        preds_k /= np.clip(preds_k.sum(axis=1).reshape(-1, 1), eps, np.inf)
        actual_k = actual[:, this_class_indices]
        # shrink predictions so
        y_hats = np.clip(preds_k, eps, 1 - eps)
        # numerator: sum  y * log(yhat)
        sum_logs = np.sum(actual_k * np.log(y_hats))
        # mean: divide numerator by N
        class_scores[k] = (-1.0 / actual.shape[0]) * sum_logs
    return np.average(class_scores) if averaged else class_scores



def log_loss_report_part(classifier, X, y, labels, class_column_indices, part="Training"):
    t=time()
    print(part + ' report started on {}'.format(datetime.now().isoformat(timespec='minutes')))
    print(part + ' accuracy         : {:.4f}'.format(classifier.score(X, y)))

    logloss = multi_multi_log_loss(y, classifier.predict_proba(X), class_column_indices, averaged=False)
    print(part + ' log loss         : {:.4f}'.format(logloss.mean()))

    print(part + ' log loss by label:')
    logloss = pd.DataFrame({'classes': [len(x) for x in class_column_indices], 'log_loss': logloss}, index=labels)
    logloss['avg_log_loss'] = logloss.log_loss / logloss.classes
    display(logloss.sort_values('avg_log_loss', ascending=False))

    logloss = multi_multi_log_loss(y, classifier.predict_proba(X), averaged=False)
    print(part + ' log loss by class:')

    logloss = pd.DataFrame({'occurrences': y.sum(), 'log_loss': logloss}, index=y.columns)
    logloss['avg_log_loss'] = logloss.log_loss * logloss.occurrences
    with pd.option_context("display.max_rows", y.shape[1]):
        display(logloss.sort_values('avg_log_loss', ascending=False))

    print(part + ' classification Report:')
    report = pd.DataFrame(
        classification_report(y, classifier.predict(X), target_names=y.columns, output_dict=True)).transpose()
    report, summary = report[:-4].sort_values('f1-score', ascending=True), report[-4:]
    with pd.option_context("display.max_rows", report.shape[0]):
        display(report)
    display(summary)
    print(part + ' report finished on {}, elapsed {:.1f} minutes\n'.format(datetime.now().isoformat(timespec='minutes'), (time()-t)/60))

def log_loss_report(classifier, X_train, y_train, X_test, y_test, labels, class_column_indices):
    log_loss_report_part(classifier, X_train, y_train, labels, class_column_indices, 'Training')
    log_loss_report_part(classifier, X_test, y_test, labels, class_column_indices, 'Testing')
