import numpy as np
import psutil
from matrix.npinfo import npinfo


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
