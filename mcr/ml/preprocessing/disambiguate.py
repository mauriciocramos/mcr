import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix


def disambiguate(df=None, reference=None, ambiguous=None, downcast=False, sparse=False):
    """
    Feature generation function which disambiguate two columns based on their missing relationship. It moves values
    from the ambiguous column to a new column based on whether the reference column is not missing or zero. After
    moving the value, the original columns is filled with a missing value NaN. Finally, those column types are
    downcast to the smallest float possible to reduce memory usage.

    :param df: a dataframe with two columns
    :param reference: the reference column name
    :param ambiguous: the ambiguous column name
    :param downcast: try to downcast float values
    :param sparse: True turns the output dataframe into a sparse matrix, Default is False
    :return: a dataframe or sparse matrix with the new disambiguation column
    """

    df = df[[reference, ambiguous]].copy()
    new_field = ambiguous+'_'+reference
    
    # df.loc[df[reference].notnull(), new_field] = df[ambiguous]
    df.loc[df[reference].notnull() & (df[reference] != 0) & df[ambiguous].notnull(), new_field] = df[ambiguous]

    # df.loc[df[reference].notnull(), ambiguous] = np.nan
    df.loc[df[reference].notnull() & (df[reference] != 0) & df[ambiguous].notnull(), ambiguous] = np.nan

    if downcast:
        df = df.apply(pd.to_numeric, downcast='float')

    return csr_matrix(df) if sparse else df
