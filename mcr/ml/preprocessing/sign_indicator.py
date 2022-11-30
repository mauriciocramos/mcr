def sign_indicator(df, missing_indicator=False):
    df = df.copy()
    cols = df.columns.tolist()
    for col in cols:
        if missing_indicator:
            df[f'{col}_isnull'] = df[col].isnull()
        df[f'{col}_negative'] = df[col] < 0
        df[f'{col}_zero'] = df[col] == 0
        df[f'{col}_positive'] = df[col] > 0
    return df.drop(columns=cols)
