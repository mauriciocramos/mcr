from size import size


def dense_matrix_report(m):
    print('Dimensions         :', m.shape)
    nnz = (m != 0).sum()
    print('Number of non-zeros:', nnz)
    print('Sparsity           :', 1 - nnz / (m.shape[0]*m.shape[1]))
    print('Data type          :', m.dtype)
    print('Size               :', size(m.nbytes))
    print(m)
