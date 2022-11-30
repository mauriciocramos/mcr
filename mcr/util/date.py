import locale
from datetime import datetime, timedelta
from matplotlib import pyplot as plt

from mcr.util import get_columns_by_content_pattern

DATE_REGEX = r'^[0-9]{4}-[0-9]{2}-[0-9]{2}'
DATETIME_REGEX = r'^[0-9]{4}-[0-9]{2}-[0-9]{2} [0-9]{2}:[0-9]{2}'


def get_date_columns(df, dtypes=('object', 'datetime64[ns]'), pattern=DATE_REGEX):
    return get_columns_by_content_pattern(df, dtypes=dtypes, pattern=pattern)


def date_timeline(df, key, date_columns=None, ascending=True):
    # Function to extract the date filling statistics of a dataframe
    if date_columns is None:
        date_columns = df.select_dtypes('datetime64[ns]').columns.tolist()
    date_flow = \
        df[date_columns].melt(ignore_index=False,
                              var_name='Field',
                              value_name='Date').sort_values([key, 'Date'], ascending=[True, ascending]).dropna()
    date_flow['Sequence'] = date_flow.groupby(level=0).transform('cumcount')
    date_flow = date_flow.pivot(columns='Sequence', values='Field').fillna('')
    date_flow = date_flow.reset_index().groupby(date_flow.columns.tolist()).nunique().sort_values(key, ascending=False)
    date_flow['%'] = date_flow[key] / date_flow[key].sum() * 100
    date_flow['acc. %'] = date_flow[key].cumsum() / date_flow[key].sum() * 100
    return date_flow


def select_datelike(df, regex=r'^[0-9]{4}-[0-9]{2}-[0-9]{2}'):
    # sub-setting fields containing dates like a REGEX
    return df.loc[:, df.apply(lambda x: x.astype('str').str.contains(regex).any())]


def date_range_plot(df, figsize=None, rot=None, threshold=None):
    plotdf = df.select_dtypes('datetime64[ns]')\
        .agg(['min', 'mean', 'max'])\
        .T\
        .sort_values(['min', 'max'])\
        .fillna(datetime.now())
    plotdf.plot(kind='bar', figsize=figsize, rot=rot)

    ymax = plotdf.max().max()
    plt.axhline(y=ymax, linestyle='--', color='red', linewidth=1)
    plt.text(y=ymax, x=-0.6, s='Furthest year found: {}'.format(ymax.year), ha='right', color='red')

    if threshold is not None:
        ythreshold = datetime.now() + timedelta(days=threshold*365)
        plt.axhline(y=ythreshold, linestyle='--', color='red', linewidth=1)
        plt.text(y=ythreshold, x=-0.6, s='{} years from now: {}'.format(threshold, ythreshold.year), ha='right',
                 color='red')

    plt.axhline(y=datetime.now(), linestyle='--', color='red', linewidth=1)
    plt.text(y=datetime.now(), x=-0.6, s='Current year: {}'.format(datetime.now().year), ha='right', color='red')

    # plt.axhline(y=datetime.fromtimestamp(0), linestyle='--', color='red', linewidth=1)
    # plt.text(y=datetime.fromtimestamp(0), x=-0.6, s='Unix Epoch: 1970', ha='right', color='red')

    ymin = plotdf.min().min()
    plt.axhline(y=ymin, linestyle='--', color='red', linewidth=1)
    plt.text(y=ymin, x=-0.6, s='Oldest year found: {}'.format(ymin.year), ha='right', color='red')

    # plt.tight_layout()
    plt.show()
    return plotdf


def date_convert(s, lc_time='pt_BR.UTF-8', cleanup='[/ ]', fmt='%B%Y'):
    current_locale = locale.getlocale(locale.LC_TIME)
    locale.setlocale(locale.LC_TIME, lc_time)
    t = s.replace(cleanup, '', regex=True)\
        .apply(lambda x: None if x == '' else datetime.strptime(x, fmt))
    locale.setlocale(locale.LC_TIME, current_locale)
    return t
