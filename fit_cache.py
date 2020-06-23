import os
import pickle
from time import time
from datetime import datetime


def fit_cache(estimator, X, y, folder, modelname):
    path = folder + modelname + '.pkl'
    if os.path.isfile(path):
        print('Loading cache {} ... '.format(modelname), end='')
        t = time()
        with open(path, 'rb') as f:
            estimator = pickle.load(f)
        print('done: {:.1f} minutes'.format((time()-t)/60))
    else:
        print('Fitting started on {}'.format(datetime.now().isoformat(timespec='minutes')))
        t = time()
        estimator.fit(X, y)
        print('Done: {:.1f} minutes'.format((time()-t)/60))
        print('Saving cache {} ... '.format(modelname), end='')
        t = time()
        with open(path, 'wb') as file:
            pickle.dump(estimator, file, pickle.HIGHEST_PROTOCOL)
        print('Done: {:.1f} minutes'.format((time()-t)/60))
    return estimator
