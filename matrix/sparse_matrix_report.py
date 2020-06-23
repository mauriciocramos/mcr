import sys
from scipy.sparse import (isspmatrix_csr, isspmatrix_csc, isspmatrix_bsr, isspmatrix_coo, isspmatrix_dok,
                          isspmatrix_dia, isspmatrix_lil)
from size import size


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
