import sys
import numpy as np
import psutil
from scipy.sparse import (isspmatrix_csr, isspmatrix_csc, isspmatrix_bsr, isspmatrix_coo, isspmatrix_dok,
                          isspmatrix_dia, isspmatrix_lil)

from mcr.util import npinfo, size


def dense_matrix(low, high=None, shape=None, sparsity=0, p=None, dtype='int64'):
    fun = {'int': np.iinfo, 'uint': np.iinfo, 'float': np.finfo}
    fun = fun[[k for k in fun.keys() if dtype.startswith(k)][0]]

    # low or high values exceed dtype min or max respectively
    assert (high-1 <= fun(dtype).max and low >= fun(dtype).min)

    area = (shape[0] * shape[1])

    # Float division adjustment options:
    if p is not None:
        length = high - low
        p = [x + (1 - sum(p)) / length for x in p]  # sum up to 1.0000000000000002 redistributing modulo
        p[-1] += 1 - sum(p)  # sum up to 1.0 by increasing the last element

    m = np.hstack((np.random.choice(np.arange(low, high, dtype=dtype), size=int(area - area * sparsity), p=p),
                   np.zeros(area - int(area - area * sparsity), dtype=dtype)))
    np.random.shuffle(m)
    return m.reshape(shape)


def dense_shape_from_memory(limit=None, dtype=None, rows=None, cols=None):
    # not both rows and cols
    assert(rows is None or cols is None)
    if limit is None:
        limit = psutil.virtual_memory().available
    elif limit <= 1:
        limit = psutil.virtual_memory().available * limit
    max_area = (limit / npinfo(dtype).dtype.itemsize)
    square_side = int(np.sqrt(max_area))
    if rows is None:
        if cols is None:
            rows, cols = square_side, square_side
        else:
            rows = np.max((1, int(max_area / cols)))
    else:
        cols = np.max((1, int(max_area / rows)))
    return (rows, cols), (1.0 * rows * cols * npinfo(dtype).dtype.itemsize)


def dense_matrix_report(m):
    print('Dimensions         :', m.shape)
    nnz = (m != 0).sum()
    print('Number of non-zeros:', nnz)
    print('Sparsity           :', 1 - nnz / (m.shape[0]*m.shape[1]))
    print('Data type          :', m.dtype)
    print('Size               :', size(m.nbytes))
    print(m)


def sparse_matrix_report(m):
    print(repr(m))
    print('Number of non-zeros  :', m.nnz)
    print('Sparsity             :', 1 - m.nnz / (m.shape[0] * m.shape[1]))
    if isspmatrix_csr(m) or isspmatrix_csc(m):
        print('data length          : {} ({})'.format(len(m.data), m.data.dtype))
        print('indptr length        : {} ({})'.format(len(m.indptr), m.indptr.dtype))
        print('indices length       : {} ({})'.format(len(m.indices), m.indices.dtype))
        print('Size                 :', size(m.data.nbytes + m.indptr.nbytes + m.indices.nbytes))
        print('10 x 10 preview:')
        print(m[:10, :10].toarray())
    elif isspmatrix_bsr(m):
        print('data length          : {} ({})'.format(len(m.data), m.data.dtype))
        print('indptr length        : {} ({})'.format(len(m.indptr), m.indptr.dtype))
        print('indices length       : {} ({})'.format(len(m.indices), m.indices.dtype))
        print('blocksize length     : {}'.format(m.blocksize))
        print('Size                 :', size(m.data.nbytes + m.indptr.nbytes + m.indices.nbytes))
        print('preview:')
        print(m)
    elif isspmatrix_coo(m):
        print('data length          : {} ({})'.format(len(m.data), m.data.dtype))
        print('row length           : {} ({})'.format(len(m.row), m.row.dtype))
        print('col length           : {} ({})'.format(len(m.col), m.col.dtype))
        print('Size                 :', size(m.data.nbytes + m.row.nbytes + m.col.nbytes))
        print('preview:')
        print(m)
    elif isspmatrix_dok(m):
        print('Size                 :', size(sys.getsizeof(m)))
        print('10 x 10 preview:')
        print(m[:10, :10].toarray())
    elif isspmatrix_dia(m):
        print('data length          : {} ({})'.format(len(m.data), m.data.dtype))
        print('Offsets              : {} ({})'.format(len(m.offsets), m.offsets.dtype))
        print('Size                 :', size(m.data.nbytes + m.offsets.nbytes))
        print('(no preview)')
    elif isspmatrix_lil(m):
        print('data length          : {} ({})'.format(len(m.data), m.data.dtype))
        print('rows                 : {} ({})'.format(len(m.rows), m.rows.dtype))
        print('Size                 :', size(m.data.nbytes + m.rows.nbytes))
        print('(no preview)')
        # print(m)
