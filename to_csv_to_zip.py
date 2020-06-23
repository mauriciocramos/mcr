import pandas as pd
from time import time
from math import floor
from zipfile import ZipFile, ZIP_DEFLATED
import os


def to_csv_to_zip(folder, basename, data, index, columns):
    print('Saving CSV...', end='')
    t = time()
    pd.DataFrame(data=data, index=index, columns=columns).to_csv(folder + basename + '.csv')
    print('Done: {:.1f} minutes'.format(floor(time()-t)/60))
    print('Zipping...', end='')
    t = time()
    with ZipFile(file=folder + basename + '.zip', mode='w', compression=ZIP_DEFLATED) as zipObj:
        zipObj.write(folder + basename + '.csv', basename + '.csv')
    os.remove(folder + basename + '.csv')
    print('Done: {:.1f} minutes'.format(floor(time()-t)/60))
