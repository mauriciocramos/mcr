import numpy as np
import pandas as pd


def get_normalized_total(df, reference, ambiguous):
    """
    Feature generation function which generates missing data binary indicators for two numeric features, and moves
    values from an ambiguous column to a new column, based on whether a reference column is not missing or
    is different than zero.

    The missing data binary indicators are named with the former reference and ambiguous column names appended with the
    prefix '_missing'.

    The ambiguous values related to the non-zero reference values are moved to a new column appended with the reference
    column name.  After moving the value, the original columns is filled with a missing value NaN.

    All resulting missing values in the 'reference', 'ambiguous' and 'ambiguous_reference' columns are imputed with zero
    with the intent to promote sparsity.  Finally, those column types are downcast to the smallest float possible.

    :param df: a dataframe
    :param reference: the reference column
    :param ambiguous: the ambiguous columns
    :return: a dataframe (copy) with the three new columns and the moved/imputed values
    """
    df = df[[reference, ambiguous]].copy()
    df[reference+'_missing'] = df[reference].isnull().astype('uint8')
    df[ambiguous+'_missing'] = df[ambiguous].isnull().astype('uint8')
    ambiguous_reference = ambiguous+'_'+reference
    df[ambiguous_reference] = np.nan
    df.loc[df[reference].notnull() & df[reference] != 0, ambiguous_reference] = df[ambiguous]
    df.loc[df[reference].notnull() & df[reference] != 0, ambiguous] = np.nan
    df[[reference, ambiguous, ambiguous_reference]] = \
        df[[reference, ambiguous, ambiguous_reference]].apply(pd.to_numeric, downcast='float')
    return df
