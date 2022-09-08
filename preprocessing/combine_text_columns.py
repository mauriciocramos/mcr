def combine_text_columns(df=None, to_drop=None, sep=' '):
    """ Takes the dataset as read in, drops the non-text columns by default and
        then combines all of the text columns into a single vector that has all of
        the text for a row.

        :param data_frame: The data as read in with read_csv (no preprocessing necessary)
        :param to_drop: (optional) Removes the numeric columns by default.
    """
    # drop non-text columns that are in the df
    if to_drop is None:
        to_drop = df.columns[df.dtypes != 'object']
    to_drop = set(to_drop) & set(df.columns)
    return df.drop(to_drop, axis=1).apply(lambda x: x.str.cat(sep=sep), axis=1)