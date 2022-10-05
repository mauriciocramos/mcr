import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix

def disambiguate(df=None, reference=None, ambiguous=None, sparse=False):
    """
    Feature generation function which disambiguate two columns based on their missing relationship.
    
    It moves values from the `ambiguous``column to a new column named `ambiguous`'_'+`reference`, based on
    whether the `reference` column is not missing or is different than zero.

    After moving the value, the original columns is filled with a missing value NaN.

    Finally, those column types are downcast to the smallest float possible to reduce memory usage.

    :param df: a dataframe
    :param reference: the reference column
    :param ambiguous: the ambiguous column
    :return: a dataframe (copy) with the three new columns and the moved/imputed values
    """
    df=df[[reference, ambiguous]].copy()
    ambiguous_reference = ambiguous+'_'+reference
    
    # df.loc[df[reference].notnull() & (df[reference]!=0) & df[ambiguous].notnull(), ambiguous_reference] = df[ambiguous]
    df.loc[df[reference].notnull(), ambiguous_reference] = df[ambiguous]
   
    # df.loc[df[reference].notnull() & (df[reference]!=0) & df[ambiguous].notnull(), ambiguous] = np.nan
    df.loc[df[reference].notnull(), ambiguous] = np.nan
    
    df = df.apply(pd.to_numeric, downcast='float')
    return csr_matrix(df) if sparse else df
