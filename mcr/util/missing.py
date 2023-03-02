from matplotlib import pyplot as plt


def plot_missing(df, figsize=None, xlim=None, decimals=2):
    # TODO: stack column and overall bars annotating only the count
    missing = missing_report(df, ascending=True)
    if figsize is None:
        figsize = (10, missing.shape[0]/4)
    if xlim is None:
        xlim = [0, missing['missing'].max()*1.3]
    ax = missing['missing'].plot(kind='barh', figsize=figsize, xlim=xlim)
    labels = [f"{row[0]:,.0f} p={row[1]*100:.{decimals}f}% ovr={row[2]*100:.{decimals}f}%" for row in missing.values]
    ax.bar_label(ax.containers[0], labels=labels)
    plt.show()


def missing_report(df, ascending=False):
    area = (df.shape[0]*(df.shape[1]))
    missing = df.isnull().agg(['sum', 'mean', lambda x: x.sum() / area]).transpose()
    missing = missing.sort_values('sum', ascending=ascending)
    missing.columns = ['missing', 'column proportion', 'overall proportion']
    missing['missing'] = missing.missing.astype('int')
    return missing
