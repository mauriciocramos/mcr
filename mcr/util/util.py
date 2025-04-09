import re
from math import pow, log, radians, sin, cos, atan2, sqrt
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display, HTML
import seaborn as sns
from scipy.stats import truncnorm, zscore
from time import time
from math import floor
from zipfile import ZipFile, ZIP_DEFLATED
import os
from PIL import Image
from glob import glob

# Canada postal code regular expression based on:
# https://www.oreilly.com/library/view/regular-expressions-cookbook/9781449327453/ch04s15.html
# r'^(?!.*[DFIOQUdfioqu])[A-VXYa-vxy][0-9][A-Za-z] ?[0-9][A-Za-z][0-9]$'
# With adjustments based on:
# https://en.wikipedia.org/wiki/Postal_codes_in_Canada
CA_POSTALCODE_REGEX = r'^(?!.*[DFIOQUdfioqu])[ABCEGHJKLMNPRSTVXYabceghjklmnprstvxy][0-9][A-Za-z] ?[0-9][A-Za-z][0-9]$'
BR_POSTALCODE_REGEX = r'^(\d){2}\.?(\d){3}-?(\d){3}$'


def get_columns_by_content_pattern(df, dtypes='object', pattern=None):
    columns_by_content_pattern = df.select_dtypes(dtypes).apply(lambda x: x.astype('str').str.contains(pattern)).any()
    return columns_by_content_pattern[columns_by_content_pattern].index.tolist()


def get_postalcode_columns(df, dtypes='object', pattern=BR_POSTALCODE_REGEX):
    return get_columns_by_content_pattern(df, dtypes=dtypes, pattern=pattern)


def head_tail(x, n=5):
    return np.concatenate((x[0:n], ['...'], x[n:][-n:])).tolist() if (len(x) > n * 2) & (n != 0) else x


def glimpse(df, n=5, info=True, deep=True):
    # Returns dataframe fields with non-null count, missing ratiod, data type, unique values and unique sample
    # Usage:
    #   with pd.option_context('display.max_rows', 500):
    #       print(glimpse(dataframe))
    if info:
        df.info(verbose=False, memory_usage='deep' if deep else True)
    g = df.apply(lambda x: (x.count(), x.isnull().mean(), x.dtype, x.nunique(),
                            head_tail(x.dropna().astype(str).sort_values().unique(), n=n)))
    g = g.T.sort_index().reset_index().rename(columns={'index': 'field', 0: 'non-null count', 1: 'missing ratio',
                                                       2: 'data type', 3: 'unique count', 4: 'unique preview'})
    return g


def multiple_replace(replace_dict, text):
    # ref: https://stackoverflow.com/questions/15175142/how-can-i-do-multiple-substitutions-using-regex
    """
    Perform multiple replacements in a string
    multiple_replace({'one': '1', 'two': '2'}, "one, two"}
    '1, 2'
    """
    # Create a regular expression  from the dictionary keys
    regex = re.compile("(%s)" % "|".join(map(re.escape, replace_dict.keys())))
    # For each match, look-up corresponding value in dictionary
    return regex.sub(lambda mo: replace_dict[mo.string[mo.start():mo.end()]], text)


def get_regex_keywords(kw_dict, text, sep=','):
    """
    Extract unique keywords from a multiple regex dictionary

    kw_dict = {r'h[íiy]brid[oa]?|semi[- ]?presencial': 'hybrid',
               r'off[- ]?site|off[- ]?pre[mise]+|remot[eoa]': 'remote',
               r'on[- ]?site|on[- ]?pre[mise]+|presencial': 'onsite'}

    get_keywords(kw_dict, text)
    """
    # extract every acceptable keyword
    kw_regex = re.compile('|'.join(kw_dict.keys()), flags=re.IGNORECASE)
    unique_kw = sorted(set(kw_regex.findall(text.lower())))
    # consolidate keywords
    consolidated_kw = []
    for item in unique_kw:
        for pattern, repl in kw_dict.items():
            match = re.search(pattern, item)
            if match:
                consolidated_kw.append(repl)
                break
    consolidated_kw = sorted(set(consolidated_kw))
    consolidated_kw = sep.join(consolidated_kw)
    return consolidated_kw if consolidated_kw != '' else None


def get_regex_keywords_from_pandas(df, kw_dict, empty='unknown'):
    s = df\
        .apply(lambda x: x.apply(lambda text: get_regex_keywords(kw_dict, text)), axis=0)\
        .apply(lambda x: ','.join(k for k in list(sorted(set(','.join(x.dropna().sort_values().unique()).split(','))))), axis=1)\
        .replace('', empty)
    return s


def get_dict_keys_types(dictionary, parent_key=None):
    # Generator function to recursively extract dictionary keys and types
    for key, value in dictionary.items():
        path = key if parent_key is None else parent_key + '.' + key
        if isinstance(value, dict):
            yield from get_dict_keys_types(value, path)
        else:
            yield path, value.__class__.__name__


def get_dict_keys(dictionary, parent_key=None):
    # Generator function to recursively extract dictionary keys
    for key, value in dictionary.items():
        path = key if parent_key is None else parent_key + '.' + key
        if isinstance(value, dict):
            yield from get_dict_keys(value, path)
        else:
            yield path


# def get_dict_keys2(dictionary, parent_key=None):
#     # Generator function to recursively extract dictionary keys
#     for key, value in dictionary.items():
#         path = key if parent_key is None else parent_key + '.' + key
#         if isinstance(value, dict):
#             yield from get_dict_keys2(value, path)
#         elif isinstance(value, list):
#             # print('path {} value {}'.format(path, value))
#             for i, v in enumerate(value):
#                 if isinstance(v, dict):
#                     # print('\tpath {} value {}'.format(path + '.' + str(i), v))
#                     yield from get_dict_keys2(v, path + '.' + str(i))
#                 else:
#                     # print(path + '.' + str(i))
#                     yield path + '.' + str(i)
#         else:
#             # print(path)
#             yield path


def sabbath(df, cols, flags, aggregations, names, swaplevel=True):
    # apply flag functions over specified columns than apply aggregations over the flags
    dfs = \
        [df[cols].apply(fun).agg(aggregations).set_axis(pd.MultiIndex.from_product([[label], names],
                                                                                   names=['flags', 'aggregations']))
         for (label, fun) in flags.items()]
    dfs = pd.concat(dfs)
    if ~swaplevel:
        dfs = dfs.swaplevel().sort_index()
    dfs = dfs.T.rename_axis('columns')
    return dfs


def pdfcdf(df, col, dropna=False):
    # Ref: https://stackoverflow.com/questions/25577352/plotting-cdf-of-a-pandas-series-in-python
    # Frequency using size rather than count to make missing frequency possible
    stats_df = df.groupby(col, dropna=dropna)[col].agg('size').pipe(pd.DataFrame).rename(columns={col: 'frequency'})
    # PDF
    stats_df['pdf'] = stats_df['frequency'] / stats_df['frequency'].sum()
    # CDF
    stats_df['cdf'] = stats_df['pdf'].cumsum()
    stats_df = stats_df.reset_index()
    return stats_df


def ecdf(data):
    """
    Compute ECDF for a one-dimensional array of measurements.
    Reference: https://www.datacamp.com/courses/statistical-thinking-in-python-part-2
    """
    # Number of data points: n
    n = len(data)
    # x-data for the ECDF: x
    x = np.sort(data)
    # y-data for the ECDF: y
    y = np.arange(1, n+1) / n
    return x, y


def groupby_percentage(df, by, column, agg, dropna=False):
    df = df.groupby(by, dropna=dropna)[column].agg(agg).to_frame().add_suffix(' ' + agg)
    df["%"] = 100 * df / df.sum()
    return df  # .sort_values('%', ascending=False)


def numeric_statistics(df, percentiles=None):
    df = df.select_dtypes('number')
    return df \
        .describe(percentiles=percentiles) \
        .T.assign(missing=df.isnull().sum(),
                  miss_ratio=df.isnull().mean(),
                  sum=df.sum(),
                  range=lambda x: x['max'] - x['min'],
                  skewness=df.skew(),
                  kurtosis=df.kurtosis(),
                  # zscore = lambda x: (x['mean'] / x['std'].replace(0, np.nan)),
                  iqr=lambda x: (x['75%'] - x['25%']),
                  lo_outlier=lambda x: (x['25%'] - (x['75%'] - x['25%']) * 1.5),
                  hi_outlier=lambda x: (x['75%'] + (x['75%'] - x['25%']) * 1.5)) \
        .astype({'count': 'int'})  # .style.format('{:,.2f}').format({'count': '{:,d}', 'missing': '{:,d}'})


# def discriminant_fillna(df, by, columns, fillna_table):
#     # Discriminant imputation function
#     # discriminant_fillna(df, by='col-A)', columns=['col-B', 'col-C'], fillna_table=fillna_table)
#     for discriminant in df[by].unique():
#         condition = df[by].isnull() if discriminant != discriminant else df[by] == discriminant
#         df.loc[condition, columns] = df.loc[condition, columns].fillna(dict(fillna_table.loc[discriminant, columns]))
#     return df


# def discriminant_sum(df, by, columns, composition_table):
#     # Discriminant sum of any components based on a composition table
#     for discriminant in df[by].unique():
#         condition = df[by].isnull() if discriminant != discriminant else df[by] == discriminant
#         df.loc[condition, columns] = df.loc[condition, columns].fillna(
#             dict(composition_table.loc[discriminant, columns]))
#     return df


# def flagsum(columns, flags):
#     # Sum any number of columns depending on respective flags
#     # Usage: df.apply(lambda row: flagsum(row[['col1', 'col2', ...]], row[['flag1', 'flag2', ...]]))
#     if not flags.any() | columns.isnull().all():
#         return np.nan
#     y = np.nan
#     for i in range(len(columns)):
#         if (columns[i] == columns[i]) & flags[i]:
#             if y != y:
#                 y = columns[i]
#             else:
#                 y += columns[i]
#     return y


def groupby_binning_plot(df, by, values, bins=(-np.inf, -1, 0, 1, np.inf), labels=('<=-1', '<=0', '<=1', '>1'),
                         layout=None,  figsize=None, right=False, subplots=True, rot=0):
    """
    Bin dataframe columns `values` as `bins`, group by columns `by` and plot bars with bins labeled as `labels`
    :param df: a dataframe
    :param by: columns to group by
    :param values: dataframe columns to bin
    :param bins: a list of bins
    :param labels: bin labels
    :param layout: (row, col) tuple.  If None, assumes number of `values`
    :param figsize: (width, height) tuple.
    :param right: Indicates whether `bins` includes the rightmost edge or not. If `right == True` (the default),
    then the `bins` `(-np.inf, -1, 0, 1, np.inf)` indicate (-inf, -1], (-1,0], (0,1], (1, inf].
    :param subplots:
    :param rot: label rotation
    :return: None
    """
    for name, group in df.groupby(by, dropna=False):
        if layout is None:
            layout = (1, len(values))
        title = '\n'.join([x + ': ' + y for x, y in list(zip(by, name))])  # +' ({} rows)'.format(group.shape[0])
        group[values] \
            .apply(lambda x: pd.cut(x, bins=bins, right=right, labels=labels).value_counts(sort=False)) \
            .plot(kind='bar', title=title, subplots=subplots, layout=layout, figsize=figsize, sharey=True, sharex=True,
                  rot=rot)
        plt.tight_layout()
        plt.show()
        print(group[values].describe(percentiles=[0.5]).T)


def trunc_norm_dist(lower, upper, mu, sigma, n):
    # Truncated normal distribution generation helper
    return truncnorm((lower - mu) / sigma, (upper - mu) / sigma, loc=mu, scale=sigma).rvs(n)


def haversine_distance(origin, destination):
    lat1, lon1 = origin
    lat2, lon2 = destination
    radius = 6371.009  # km
    dlat = radians(lat2 - lat1)
    dlon = radians(lon2 - lon1)
    a = (sin(dlat / 2) * sin(dlat / 2) +
         cos(radians(lat1)) * cos(radians(lat2)) *
         sin(dlon / 2) * sin(dlon / 2))
    c = 2 * atan2(sqrt(a), sqrt(1 - a))
    d = radius * c
    return d


def play(src, auto=False):
    # display audio player from a file in the Juypter Notebook's current folder
    display(HTML(f'<audio controls{" autoplay" if auto else ""} src="{src}" type="mpeg/audio"></audio>'))


def correlation_heatmap(cm, title=None, figsize=None):
    mask = np.triu(np.ones_like(cm, dtype=bool))  # mask for the upper triangle
    plt.figure(figsize=figsize)
    heatmap = sns.heatmap(cm, mask=mask, annot=True, vmin=0, vmax=1)
    heatmap.set_title(title)  # ,fontdict={'fontsize':12}, pad=12);
    plt.xticks(rotation=45)


def binary_pivot_table(df, index_pattern, value_pattern):
    # df should be a dataframe of binary values

    pvt = df.astype('uint8').pivot_table(index=df.columns[df.columns.str.match(index_pattern)].tolist(),
                                         values=df.columns[df.columns.str.match(value_pattern)].tolist(),
                                         aggfunc=['sum'])
    # display(pvt.style.bar(axis=None, color='blue'))
    cmap = sns.dark_palette("blue", as_cmap=True)
    display(pvt.style.background_gradient(axis=None, cmap=cmap))

    pvt = df.astype('uint8').pivot_table(index=df.columns[df.columns.str.match(index_pattern)].tolist(),
                                         values=df.columns[df.columns.str.match(value_pattern)].tolist(),
                                         aggfunc=['mean'])
    # display(pvt.style.bar(axis=None, color='green'))
    cmap = sns.dark_palette("green", as_cmap=True)
    display(pvt.style.background_gradient(axis=None, cmap=cmap))


def rolling_plot(s, resample, rolling, title, figsize=(24, 5)):
    resample, resample_func = resample
    rolling, rolling_func = rolling
    rolling_window = str(rolling * int(re.sub(r'\D', '', resample))) + re.sub(r'\d', '', resample)
    ylabel = s.name
    linestyle = '-'

    resample_agg = s.resample(resample).agg(resample_func)
    resample_agg.plot(figsize=figsize, linestyle=linestyle, label=f'{resample} {resample_func}', color='blue')

    rolling_agg = resample_agg.rolling(rolling).agg(rolling_func)
    rolling_agg.plot(figsize=figsize, marker='', linestyle=linestyle, label=f'{rolling_window} rolling {rolling_func}',
                     color='red', title=title)

    plt.ylabel(ylabel)
    plt.legend()


def size(size_bytes):
    if size_bytes < 1024:
        return '{} B'.format(size_bytes)
    size_name = ("B", "KiB", "MiB", "GiB", "TiB", "PiB", "EiB", "ZiB", "YiB")
    i = int(floor(log(size_bytes, 1024)))
    p = pow(1024, i)
    s = size_bytes / p  # s = round(size_bytes / p, 2)
    return '{:.1f} {}'.format(s, size_name[i])


def max_interactions(n, minus=0):
    # Returns the number of features whose interactions don't exceed a maximum of n features
    # minus the number of pre-existing features
    return int(((8 * n + 1)**(1/2) - 1) / 2) - minus


def to_csv_to_zip(folder, basename, data, index, columns, verbose=True, remove_csv=True):
    if verbose:
        print('Saving CSV...', end='')
    t = time()
    pd.DataFrame(data=data, index=index, columns=columns).to_csv(folder + basename + '.csv')
    if verbose:
        print(f'done in {(time()-t)/60:.1f} minute(s).')
        print('Zipping...', end='')
    t = time()
    with ZipFile(file=folder + basename + '.zip', mode='w', compression=ZIP_DEFLATED) as zipObj:
        zipObj.write(folder + basename + '.csv', basename + '.csv')
    if remove_csv:
        os.remove(folder + basename + '.csv')
    if verbose:
        print(f'done in {(time()-t)/60:.1f} minute(s).')


def npinfo(dtype):
    # unifies numpy.iinfo() and numpy.finfo()
    fun = {'int': np.iinfo, 'uint': np.iinfo, 'float': np.finfo}
    return fun[[k for k in fun.keys() if dtype.startswith(k)][0]](dtype)


def plot_unique(df, figsize=None, xlim=None, decimals=2):
    unique = pd.DataFrame({'sum': df.nunique(), 'proportion': df.nunique() / len(df)}).sort_values('sum')
    if figsize is None:
        figsize = (10, unique.shape[0]/4)
    if xlim is None:
        xlim = [0, unique['sum'].max()*1.25]
    ax = unique['sum'].plot(kind='barh', figsize=figsize, xlim=xlim)
    labels = [f"{row[0]:,.0f} ({row[1]*100:.{decimals}f}%)" for row in unique.values]
    ax.bar_label(ax.containers[0], labels=labels)


def plot_duplicates(df, figsize=None, xlim=None, decimals=2):
    duplicates = pd.DataFrame({'sum': len(df) - df.nunique(), 'proportion': 1 - df.nunique() / len(df)}).sort_values('sum')
    if figsize is None:
        figsize = (10, duplicates.shape[0]/4)
    if xlim is None:
        xlim = [0, duplicates['sum'].max()*1.25]
    ax = duplicates['sum'].plot(kind='barh', figsize=figsize, xlim=xlim)
    labels = [f"{row[0]:,.0f} ({row[1]*100:.{decimals}f}%)" for row in duplicates.values]
    ax.bar_label(ax.containers[0], labels=labels)


def plot_counts(s, figsize=None, xlim=None, decimals=2, min_frequency=1):
    s = s.sort_values(ascending=True)
    values = pd.DataFrame({'sum': s,
                           'proportion': s / s.sum()})  # .sort_values('sum')
    values = values.loc[values['sum'] >= min_frequency]
    if figsize is None:
        figsize = (19.2 / 1, 10.8 * values.shape[0] / 50)
    if xlim is None:
        xlim = [0, values['sum'].max()*1.2]
    ax = values['sum'].plot(kind='barh',
                            figsize=figsize,
                            xlim=xlim,
                            title=f'top {values.shape[0]}/{s.shape[0]}',
                            width=0.90)
    labels = [f"{row[0]:,.0f} ({row[1]*100:.{decimals}f}%)" for row in values.values]
    ax.bar_label(ax.containers[0], labels=labels, label_type='edge', color='white')


def plot_value_counts_timeseries(df, index, column, top=7, rule='D', figsize=(19.2, 10.8)):
    top_values = df[column].value_counts().head(top).index.tolist()
    df\
        .loc[df[column].isin(top_values), [index, column]].astype({column: 'object'})\
        .groupby([index, column], dropna=False)[column].count().rename(f'{column}_count')\
        .to_frame()\
        .reset_index(level=1).pivot(columns=column).droplevel(0, axis=1)\
        .resample(rule).sum()\
        .plot(figsize=(19.2, 10.8))
    plt.title(f'Top {top} {column} time series ({rule=})')


def plot_value_counts(s, figsize=None, xlim=None, decimals=2, min_frequency=1, max_frequency=1.0,
                      max_cum_frequency=1.0, max_features=None):
    assert \
        (isinstance(min_frequency, int) and 0 < min_frequency) or \
        (isinstance(min_frequency, float) and 0 < min_frequency <= 1.0), \
        'min_frequency must be a positive integer or a float between 0.0 (exclusive) and 1.0 (inclusive)'
    assert \
        (isinstance(max_frequency, int) and 0 < max_frequency) or \
        (isinstance(max_frequency, float) and 0 < max_frequency <= 1.0), \
        'max_frequency must be a positive integer or a float between 0.0 (exclusive) and 1.0 (inclusive)'
    assert \
        (isinstance(max_cum_frequency, int) and 0 < max_cum_frequency) or \
        (isinstance(max_cum_frequency, float) and 0 < max_cum_frequency <= 1.0), \
        'max_cum_frequency must be a positive integer or a float between 0.0 (exclusive) and 1.0 (inclusive)'
    assert \
        max_features is None or (isinstance(max_features, int) and max_features >= 1), \
        'max_features must be None or a positive integer'

    if s.count() == 0:
        print(f'No values found in {s.name}')
        return
    vc = s.value_counts(dropna=False)
    vp = vc / len(s)
    values = pd.DataFrame({'sum': vc,
                           'cum_sum': vc.cumsum(),
                           'proportion': vp,
                           'cum_proportion': vp.cumsum()})
    total_size = len(values)

    if isinstance(min_frequency, int) and min_frequency > 1:
        values = values.loc[(values['sum'] >= min_frequency)]
    elif isinstance(min_frequency, float):
        values = values.loc[(values['proportion'] >= min_frequency)]
    if len(values) == 0:
        print(f'No values found for {s.name}. Try to adjust min_frequency ({min_frequency})')
        return

    if isinstance(max_frequency, int):
        values = values.loc[(values['sum'] <= max_frequency)]
    elif isinstance(max_frequency, float) and max_frequency < 1.0:
        values = values.loc[(values['proportion'] <= max_frequency)]
    if len(values) == 0:
        print(f'No values found for {s.name}. Try to adjust max_frequency ({max_frequency})')
        return

    if isinstance(max_cum_frequency, int):
        values = values.loc[(values['cum_sum'] <= max_cum_frequency)]
    elif isinstance(max_cum_frequency, float) and max_cum_frequency < 1.0:
        values = values.loc[(values['cum_proportion'] <= max_cum_frequency)]
    if len(values) == 0:
        print(f'No values found for {s.name}. Try to adjust max_cum_frequency ({max_cum_frequency})')
        return

    if max_features is not None:
        values = values.head(max_features)
    if len(values) == 0:
        print(f'No values found for {s.name}. Try to adjust max_features ({max_features})')
        return

    if figsize is None:
        figsize = (19.2 / 1, min(2**16 - 1, 10.8 * len(values) / 50))
    if xlim is None:
        xlim = [0, values['sum'].max()*1.2]
    labels = [f"{row[0]:,.0f} ({row[2]*100:.{decimals}f}%)" for row in values.values]

    ax = values['sum'].plot(kind='barh',
                            figsize=figsize,
                            xlim=xlim,
                            title=f'{s.name}: top {len(values)} of {total_size}',
                            width=0.90)
    ax.bar_label(ax.containers[0], labels=labels, label_type='edge', color='white')
    plt.gca().invert_yaxis()


def density_plots(x, label=None, outlying=True, zthreshold=3, bins=None, figsize=(19, 4)):

    label = x.name if label is None else label
    if outlying:
        x_mean = x.mean()
        x_std = x.std()
        lower_limit = x_mean - zthreshold * x_std
        upper_limit = x_mean + zthreshold * x_std
        zscr = zscore(x, nan_policy='omit')
        outliers = x[(zscr < -zthreshold) | (zscr > zthreshold)]

    plt.figure(figsize=figsize)

    # scatter plot
    plt.subplot(1,3,1); plt.grid(False)
    plt.plot(x, marker='.', linestyle='none', color='green', label='inlier')
    plt.ylabel(label); plt.xlabel('row')
    if outlying:
        plt.plot(outliers, marker='.', linestyle='none', color='red', label='outlier')
        plt.axhline(lower_limit, linestyle='dotted', color='red', label=f'{zthreshold} z-score')
        plt.axhline(upper_limit, linestyle='dotted', color='red')
        plt.legend(loc='best')

    # density plot
    plt.subplot(1,3,2); plt.grid(False)
    plt.hist(x, bins=bins)
    plt.xlabel(label); plt.ylabel('PDF')

    if outlying:
        plt.axvline(lower_limit, linestyle='dotted', color='red', label=f'{zthreshold} z-score')
        plt.axvline(upper_limit, linestyle='dotted', color='red')
        plt.legend(loc='best')

    # box plot
    plt.subplot(1,3,3); plt.grid(False)
    plt.boxplot(x.dropna())
    plt.ylabel(label)

    if outlying:
        plt.axhline(lower_limit, linestyle='dotted', color='red', label=f'{zthreshold} z-score')
        plt.axhline(upper_limit, linestyle='dotted', color='red')
        plt.legend(loc='best')

    plt.tight_layout()
    plt.show()


def plot_regression_correlation_imputation(df, x, y, figsize=None):
    zero_imputed_df = df[[x, y]].fillna(0)
    mean_imputed_df = df[[x, y]].fillna(df[[x, y]].mean())
    median_imputed_df = df[[x, y]].fillna(df[[x, y]].median())

    plt.figure(figsize=figsize)
    xytext = (0.05, 0.5)
    textcoords='axes fraction'

    plt.subplot(2,2,1)
    sns.regplot(x=x, y=y, data=df[[x,y]])
    plt.annotate('Pearson = {:.12f}'.format(df[[x,y]].corr().iloc[0,1]), xy=xytext, xytext=xytext, textcoords=textcoords)
    plt.title('Raw')

    plt.subplot(2,2,2)
    sns.regplot(x=x, y=y, data=zero_imputed_df)
    plt.title('Zero imputed')
    plt.annotate('Pearson = {:.12f}'.format(zero_imputed_df[[x,y]].corr().iloc[0,1]), xy=xytext, xytext=xytext, textcoords=textcoords)

    plt.subplot(2,2,3)
    sns.regplot(x=x, y=y, data=mean_imputed_df)
    plt.title('Mean imputed')
    plt.annotate('Pearson = {:.12f}'.format(mean_imputed_df[[x,y]].corr().iloc[0,1]), xy=xytext, xytext=xytext, textcoords=textcoords)

    plt.subplot(2,2,4)
    sns.regplot(x=x, y=y, data=median_imputed_df)
    plt.title('Median imputed')
    plt.annotate('Pearson = {:.12f}'.format(median_imputed_df[[x,y]].corr().iloc[0,1]), xy=xytext, xytext=xytext, textcoords=textcoords)

    plt.tight_layout()
    plt.show()


def plot_grouped_correlation_imputation(df, x, y, by, figsize=(7, 7)):
    grouped_corr = df[[by, x, y]] \
                       .groupby(by).corr().unstack().iloc[:, 2].to_frame('raw')

    grouped_corr['zero imputed'] = df[[by, x, y]].fillna({x: 0, y: 0}) \
                                       .groupby(by).corr().unstack().iloc[:, 2].rename('zero imputed')

    grouped_corr['mean imputed'] = df[[by, x, y]].fillna({x: df[x].mean(), y: df[y].mean()}) \
                                       .groupby(by).corr().unstack().iloc[:, 2].rename('mean imputed')

    grouped_corr['median imputed'] = df[[by, x, y]].fillna({x: df[x].median(), y: df[y].median()}) \
                                         .groupby(by).corr().unstack().iloc[:, 2].rename('median imputed')

    grouped_corr.sort_values('raw', ascending=True, na_position='first').plot(kind='barh', figsize=figsize,
                                                                              xlim=(-1, 1))

    plt.title(f'{x} and {y} correlations grouped by {by}')
    plt.xlabel('Pearson Correlation')
    plt.grid(axis='x')
    plt.tight_layout()
    plt.show()


def plot_grouped_statistics(df, cols, by=None, func=('median', 'mean'), figsize=(14, 7)):
    fig, axes = plt.subplots(nrows=1, ncols=len(cols), figsize=figsize)
    fig.suptitle(f'Grouped by [{by}] subset [{", ".join(cols)}] aggregation [{", ".join(func)}]')

    df.groupby(by)[cols[0]].agg(func).plot(kind='barh', ax=axes[0], legend=False)
    fig.legend(loc='upper left')
    axes[0].set_xlabel(cols[0])

    for i in range(1, len(cols)):
        df.groupby(by)[cols[i]].agg(func).plot(kind='barh', ax=axes[i], legend=False)
        axes[i].set_xlabel(cols[i])
        axes[i].set_ylabel(None)
        axes[i].set_yticks([])

    plt.tight_layout()
    plt.show()


def cross_value_counts(df, pattern=r'\s+', lowercase=True):
    r"""
    pd.Series.values_count() wrapper applied over dataframe columns after preprocessing a configurable regular
    expression clean-up and lower case. The output is sorted by row sum and then by column sum
    :param df: dataframe of text columns
    :param pattern: clean-up regular expression pattern such as r'\[^\w\]+', default r'\s+'
    :param lowercase: whether to lowercase before counting
    :return: dataframe of value counts with values as rows and column names as columns
    """
    if lowercase:
        cvc = df.replace(pattern, ' ', regex=True).apply(lambda x: x.str.strip().str.lower())
    else:
        cvc = df.replace(pattern, ' ', regex=True).apply(lambda x: x.str.strip())
    cvc = cvc.apply(lambda x: x.value_counts(dropna=False)).fillna(0).apply(pd.to_numeric, downcast='unsigned')
    cvc.index.name = 'text'
    cvc.columns.name = 'columns'
    cvc.index = cvc.index.fillna('NaN')
    # sorting value counts by rows than by columns
    decreasing_row_index = cvc.sum(axis=1).sort_values(ascending=False).index
    decreasing_col_index = cvc.loc[cvc.index != 'NaN'].sum(axis=0).sort_values(ascending=False).index
    return cvc.loc[decreasing_row_index, decreasing_col_index]  # .style.bar()


def cross_value_counts_summary(df):
    percent_with_nan = lambda x: x.sum() / (df.shape[0] * df.shape[1]) * 100
    cvc = cross_value_counts(df).replace(0, np.nan)
    cvcs = cvc.agg(['sum', 'count', percent_with_nan], axis='columns')
    cvcs.index.name = 'text'
    cvcs.columns=['Ocurrences', 'Columns', '% w/NaN']

    saved = cvcs.loc[cvcs.index=='NaN','Ocurrences']
    cvcs.loc[cvcs.index=='NaN','Ocurrences'] = np.nan
    cvcs['% wo/NaN'] = (cvcs.Ocurrences / cvcs.Ocurrences.sum() * 100)#.fillna(0)
    cvcs['Cumulated % wo/NaN'] = cvcs['% wo/NaN'].cumsum()
    cvcs.loc[cvcs.index=='NaN','Ocurrences'] = saved
    cvcs['words'] = cvcs.index.str.split().map(len).tolist()
    cvcs.loc[cvcs.index=='NaN','words'] = np.nan

    cvcs = cvcs.apply(pd.to_numeric, downcast='float')
    cvcs = cvcs.apply(pd.to_numeric, downcast='unsigned')

    return cvcs


def image_metadata(path):
    img = Image.open(path)
    extrema = np.array(img.getextrema())
    pixel_count = img.size[0] * img.size[1]
    # getcolor returns an unsorted list of (count, pixel) values.  Pixel can be a scalar int or a tuple of 3 ints
    num_colors = len(img.getcolors(maxcolors=pixel_count))
    means = (
        np.array(
            [
                count * np.array(pixel if isinstance(pixel, list) else [pixel])
                for count, pixel in img.getcolors(maxcolors=pixel_count)
            ]
        ).sum(axis=0)
        / pixel_count
    )
    stds = (
        np.array(
            [
                (np.array(pixel if isinstance(pixel, list) else [pixel]) - means) ** 2
                for count, pixel in img.getcolors(maxcolors=pixel_count)
            ]
        ).sum(axis=0)
        / pixel_count
    ) ** (1 / 2)
    return {
        "mime_type": img.get_format_mimetype(),
        "width": img.size[0],
        "height": img.size[1],
        "bands": "".join(img.getbands()),
        "num_colors": num_colors,
        "min_color": extrema.min(),
        "max_color": extrema.max(),
        **{f"color_mean_{i}": mu for i, mu in enumerate(means)},
        **{f"color_stds_{i}": mu for i, mu in enumerate(stds)},
    }


def image_folder_metadata(pathname, sort=True, key=None, reverse=False, sep="/", columns=None):
    paths = glob(pathname)
    if sort:
        paths.sort(key=key, reverse=reverse)
    dfa = pd.DataFrame([ path.split("/") for path in paths ]).add_prefix('path_')
    # dfa.columns = [f'dir{level}' for level in range(dfa.columns.size - 1)] + ['filename']
    dfb = pd.DataFrame([image_metadata(path) for path in paths])
    return pd.concat((dfa, dfb), axis=1)