import numpy as np
import pandas as pd
from IPython.display import display
from seaborn import dark_palette


def log_loss(y_true, y_pred, eps=1e-15):
    """ Computes the logarithmic loss between y_pred and y_true when these are 1D arrays
        Reference: https://www.datacamp.com/courses/machine-learning-with-the-experts-school-budgets
        :param y_pred: The y_pred probabilities as floats between 0-1
        :param y_true: The y_true binary labels. Either 0 or 1.
        :param eps: (optional) log(0) is inf, so we need to offset our y_pred values slightly by eps from 0 or 1.

        # sklearn logloss eps: 1e-15
        # np.finfo(np.float64()).resolution: 1e-15
    """
    y_pred = np.clip(y_pred, eps, 1 - eps)
    loss = -1 * np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
    return loss


def multi_multi_log_loss(y_true, y_pred, class_column_indices=None, averaged=True, eps=1e-15):
    """ Multi class version of Logarithmic Loss metric as implemented on
        DrivenData.org
        Reference: https://github.com/datacamp/course-resources-ml-with-experts-budgets/blob/master/src/models/metrics.py
    """

    # Default flatten indices
    if class_column_indices is None:
        class_column_indices = [[i] for i in range(y_true.shape[1])]
    class_scores = np.ones(len(class_column_indices), dtype=np.float64)
    # avoids add .values in y_true upstream
    if isinstance(y_true, pd.DataFrame):
        y_true = y_true.values
    # calculate log loss for each set of columns that belong to a class:
    for k, this_class_indices in enumerate(class_column_indices):
        # get just the columns for this class
        preds_k = y_pred[:, this_class_indices].astype(np.float64)
        # normalize so probabilities sum to one (unless sum is zero, then we clip)
        preds_k /= np.clip(preds_k.sum(axis=1).reshape(-1, 1), eps, np.inf)
        actual_k = y_true[:, this_class_indices]
        # shrink predictions so
        y_hats = np.clip(preds_k, eps, 1 - eps)
        # numerator: sum  y * log(yhat)
        sum_logs = np.sum(actual_k * np.log(y_hats))
        # mean: divide numerator by N
        class_scores[k] = (-1.0 / y_true.shape[0]) * sum_logs
    return np.average(class_scores) if averaged else class_scores


def log_loss_report(classifier, X, y, labels, class_column_indices, part=None, summary=True,
                    color='red', formatter='{:.21f}'):
    print(f'{part} accuracy         : {classifier.score(X, y)}')
    classifier_predict_proba = classifier.predict_proba(X)
    logloss = multi_multi_log_loss(y, classifier_predict_proba, class_column_indices, averaged=False)
    print(f'{part} log loss         : {logloss.mean()}')
    if not summary:

        print(f'{part} log loss by label:')
        logloss = pd.DataFrame({'num_classes': [len(x) for x in class_column_indices], 'log_loss': logloss}, index=labels)
        logloss['log_loss_percent'] = logloss['log_loss'] / logloss['log_loss'].sum()
        logloss['log_loss_per_class'] = logloss['log_loss'] / logloss['num_classes']
        cmap = dark_palette(color, as_cmap=True)
        display(logloss
                .sort_values('log_loss', ascending=False)
                .style.background_gradient(cmap=cmap))

        print(f'{part} log loss by class:')
        logloss = multi_multi_log_loss(y, classifier_predict_proba, averaged=False)
        logloss = pd.DataFrame({'occurrences': y.sum(), 'log_loss': logloss}, index=y.columns)
        logloss['log_loss_per_occurrence'] = logloss['log_loss'] / logloss['occurrences']
        with pd.option_context("display.max_rows", y.shape[1]):
            # display(logloss.sort_values('log_loss_per_occurrence', ascending=False))
            display(logloss.style
                    .format(formatter, subset=['log_loss', 'log_loss_per_occurrence'])
                    .background_gradient(cmap=cmap))


def log_loss_report_complete(classifier, X_train, y_train, X_test, y_test, labels, class_column_indices, summary=True,
                             color='red', formatter='{:.21f}'):
    log_loss_report(classifier, X_train, y_train, labels, class_column_indices, part='Training', summary=summary,
                    color=color, formatter=formatter)
    log_loss_report(classifier, X_test, y_test, labels, class_column_indices, part='Testing', summary=summary,
                    color=color, formatter=formatter)
