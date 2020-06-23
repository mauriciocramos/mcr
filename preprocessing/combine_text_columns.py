def combine_text_columns(data_frame, to_drop=None):
    """ Takes the dataset as read in, drops the non-text columns by default and
        then combines all of the text columns into a single vector that has all of
        the text for a row.
        Reference: https://www.datacamp.com/courses/machine-learning-with-the-experts-school-budgets

        :param data_frame: The data as read in with read_csv (no preprocessing necessary)
        :param to_drop: (optional) Removes the numeric columns by default.
    """
    # drop non-text columns that are in the df
    if to_drop is None:
        to_drop = data_frame.columns[data_frame.dtypes != 'object']
    to_drop = set(to_drop) & set(data_frame.columns)

    text_data = data_frame.drop(to_drop, axis=1)

    # replace nans with empty string
    text_data.fillna('', inplace=True)

    # joins all of the text items in a row (axis=1)
    # with a space in between
    return text_data.apply(lambda x: ' '.join(x), axis=1)
