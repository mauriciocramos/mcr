import numpy as np
import pandas as pd


def get_normalized_total(df, reference, total):
    # df = df[['FTE', 'Total']].copy()
    # df['FTE_missing'] = df.FTE.isnull().astype('uint8')
    # df['Total_missing'] = df.Total.isnull().astype('uint8')
    # df['Total_FTE'] = np.nan
    # df.loc[df.FTE.notnull() & df.FTE != 0, 'Total_FTE'] = df.Total
    # df.loc[df.FTE.notnull() & df.FTE != 0, 'Total'] = np.nan
    # df[['FTE', 'Total', 'Total_FTE']] = df[['FTE', 'Total', 'Total_FTE']].apply(pd.to_numeric, downcast='float')
    # return df
    df = df[[reference, total]].copy()
    df[reference+'_missing'] = df[reference].isnull().astype('uint8')
    df[total+'_missing'] = df[total].isnull().astype('uint8')
    total_reference = total+'_'+reference
    df[total_reference] = np.nan
    df.loc[df[reference].notnull() & df[reference] != 0, total_reference] = df[total]
    df.loc[df[reference].notnull() & df[reference] != 0, total] = np.nan
    df[[reference, total, total_reference]] = df[[reference, total, total_reference]].apply(pd.to_numeric, downcast='float')
    return df