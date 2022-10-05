import numpy as np


def npinfo(dtype):
    # unifies numpy.iinfo() and numpy.finfo()
    fun = {'int': np.iinfo, 'uint': np.iinfo, 'float': np.finfo}
    return fun[[k for k in fun.keys() if dtype.startswith(k)][0]](dtype)
