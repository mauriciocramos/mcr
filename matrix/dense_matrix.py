import numpy as np


def dense_matrix(low, high=None, size=None, sparsity=0, p=None, dtype='int64'):
    fun = {'int': np.iinfo, 'uint': np.iinfo, 'float': np.finfo}
    fun = fun[[k for k in fun.keys() if dtype.startswith(k)][0]]

    # low or high values exceed dtype min or max respectively
    assert (high-1 <= fun(dtype).max and low >= fun(dtype).min)

    area = (size[0] * size[1])

    # Float division adjustment options:
    if p is not None:
        length = high - low
        p = [x + (1 - sum(p)) / length for x in p]  # sum up to 1.0000000000000002 redistributing modulo
        p[-1] += 1 - sum(p)  # sum up to 1.0 by increasing the last element

    m = np.hstack((np.random.choice(np.arange(low, high, dtype=dtype), size=int(area - area * sparsity), p=p),
                   np.zeros(area - int(area - area * sparsity), dtype=dtype)))
    np.random.shuffle(m)
    return m.reshape(size)
