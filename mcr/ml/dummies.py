from collections import defaultdict
import pandas as pd
import numpy as np


def reverse_dummies(df_dummies, prefix_sep='_'):
    """
    Reference: https://stackoverflow.com/questions/34523111/the-most-elegant-way-to-get-back-from-pandas-df-dummies

    :param df_dummies:
    :param prefix_sep:
    :return:
    """
    pos = defaultdict(list)
    vals = defaultdict(list)
    for i, c in enumerate(df_dummies.columns):
        if "_" in c:
            k, v = c.split(prefix_sep, 1)
            pos[k].append(i)
            vals[k].append(v)
        else:
            pos[prefix_sep].append(i)
    df = pd.DataFrame({k: pd.Categorical.from_codes(
                              np.argmax(df_dummies.iloc[:, pos[k]].values, axis=1),
                              vals[k])
                      for k in vals})
    df.index=df_dummies.index
    df[df_dummies.columns[pos[prefix_sep]]] = df_dummies.iloc[:, pos[prefix_sep]]
    return df


def get_dummies_indices(df):
    label_margins = df.apply(pd.Series.nunique).cumsum()
    return [list(range(start, stop)) for start, stop in zip(np.concatenate(([0], label_margins[:-1])), label_margins)]
