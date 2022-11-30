import os
import pickle
from time import time
from datetime import datetime


def fit_cache(estimator, X, y, folder, modelname):
    path = folder + modelname + '.pkl'
    if os.path.isfile(path):
        print(f'Loading cache {modelname} ... ', end='')
        t = time()
        with open(path, 'rb') as f:
            estimator = pickle.load(f)
        print(f'done: {(time()-t)/60:.1f} minutes')
    else:
        print(f'Fitting started on {datetime.now().isoformat(timespec="minutes")}')
        t = time()
        estimator.fit(X, y)
        print(f'Done: {(time()-t)/60:.1f} minutes')
        print(f'Saving cache {modelname} ... ', end='')
        t = time()
        with open(path, 'wb') as file:
            pickle.dump(estimator, file, pickle.HIGHEST_PROTOCOL)
        print(f'Done: {(time()-t)/60:.1f} minutes')
    return estimator


def pickle_cache(x, file_name):
    path = file_name + '.pkl'
    if os.path.isfile(path):
        with open(path, 'rb') as f:
            x = pickle.load(f)
    else:
        with open(path, 'wb') as file:
            pickle.dump(x, file, pickle.HIGHEST_PROTOCOL)
    return x
