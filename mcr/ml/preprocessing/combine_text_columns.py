def combine_text_columns(df, to_drop=None, sep=' '):
    """Takes the dataset as read in, drops the non-text columns by default and then combines all the text columns into
    a single vector that has all the text for a row.

        :param df: The data as read in with read_csv (no preprocessing necessary)
        :param to_drop: (optional) Removes the numeric columns by default.
        :param sep: Default ' '. Could be '. ' for sentence tokenizing.
    """
   
    if to_drop is None:
        # drop non-text columns that are in the df
        to_drop = df.columns[df.dtypes != 'object']
    else:
        # drop selected plus non-object in case user forgot to include
        to_drop = (set(to_drop) & set(df.columns)) | set(df.columns[df.dtypes != 'object'])
    
    return df\
        .drop(to_drop, axis=1)\
        .replace(r'\s+', ' ', regex=True)\
        .apply(lambda x: x.str.strip())\
        .apply(lambda x: x.str.cat(sep=sep), axis=1) + sep.strip()
