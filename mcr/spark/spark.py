import datetime
import numpy as np
from pyspark.sql import functions as F


def min_max_scaler(df, cols_to_scale):
    # Takes a dataframe and list of columns to minmax scale. Returns a dataframe.
    for col in cols_to_scale:
        # Define min and max values and collect them
        max_days = df.agg({col: 'max'}).collect()[0][0]
        min_days = df.agg({col: 'min'}).collect()[0][0]
        new_column_name = 'scaled_' + col
        # Create a new column based off the scaled data
        df = df.withColumn(new_column_name,
                           (df[col] - min_days) / (max_days - min_days))
    return df


def column_dropper(df, threshold):
    # Takes a dataframe and threshold for missing values. Returns a dataframe.
    total_records = df.count()
    for col in df.columns:
        # Calculate the percentage of missing values
        missing = df.where(df[col].isNull()).count()
        missing_percent = missing / total_records
        # Drop column if percent of missing is more than threshold
        if missing_percent > threshold:
            df = df.drop(col)
    return df


def drop_low_observation_columns(df, columns, threshold=30):
    return df.drop(*np.array(columns)[np.array([df.agg({f'`{col}`': 'sum'}).collect()[0][0] < threshold
                                                for col in columns])])


def train_test_split_date(df, split_col, test_days):
    split_date = None
    if isinstance(test_days, float):
        max_date = df.agg({split_col: 'max'}).collect()[0][0]
        min_date = df.agg({split_col: 'min'}).collect()[0][0]
        split_in_days = int((max_date - min_date).days * 0.8)
        split_date = min_date + datetime.timedelta(days=split_in_days)
    elif isinstance(test_days, int):
        max_date = df.agg({split_col: 'max'}).collect()[0][0]
        split_date = max_date - datetime.timedelta(days=test_days)
    return split_date


def split_explode_join(df, groupby, column, sep=', ', concat_sep='_'):
    exploded_df = df[[groupby, column]]\
        .fillna('NaN', subset=column)\
        .withColumn(f'{column}_LIST', F.split(F.upper(column), sep))\
        .withColumn(f'EXPLODED_{column}_LIST', F.explode(f'{column}_LIST'))\
        .withColumn(f'EXPLODED_{column}_LIST', F.concat(F.lit(f'{column}{concat_sep}'),
                                                        F.trim(f'EXPLODED_{column}_LIST')))\
        .withColumn('ONE', F.lit(1))\
        .groupBy(groupby).pivot(f'EXPLODED_{column}_LIST').agg(F.coalesce(F.first('ONE')))
    return df.join(exploded_df, on=groupby, how='left')


def prefixed_join(df, groupby, column, concat_sep=':'):
    prefixed_df = df[[groupby, column]]\
        .dropna(subset=column)\
        .withColumn(f'{column.upper()}_PREFIXED', F.concat(F.lit(f'{column}{concat_sep}'), F.upper(F.trim(column))))\
        .withColumn('ONE', F.lit(1))\
        .groupBy(groupby).pivot(f'{column}_PREFIXED').agg(F.coalesce(F.first('ONE')))
    return df.join(prefixed_df, on=groupby, how='left')


def roem(spark, predictions, user_col=None, rating_col=None):
    """
    (ROEM) Rank Ordering Error Metric for Spark ALS implicit ratings models
    Expected percentile rank error metric function
    https://github.com/jamenlong/ALS_expected_percent_rank_cv/blob/master/ROEM_cv.py
    """
    # Creates table that can be queried
    predictions.createOrReplaceTempView("predictions")
    # Sum of total number of plays of all songs
    denominator = predictions.groupBy().sum(rating_col).collect()[0][0]
    # Calculating rankings of songs predictions by user
    spark.sql("SELECT " + user_col + " , " + rating_col + " , PERCENT_RANK() OVER (PARTITION BY " + user_col +
              " ORDER BY prediction DESC) AS rank FROM predictions").createOrReplaceTempView("rankings")
    # Multiplies the rank of each song by the number of plays and adds the products together
    numerator = spark.sql('SELECT SUM(' + rating_col + ' * rank) FROM rankings').collect()[0][0]
    performance = numerator / denominator
    return performance
