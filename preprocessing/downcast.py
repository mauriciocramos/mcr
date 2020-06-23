def downcast(x, desired=None):
    types = {'float': ['float64', 'float32'],
             'integer': ['int64', 'int32', 'int16', 'int8'],
             'unsigned': ['uint64', 'uint32', 'uint16', 'uint8']}
    # Only types float, integer and unsigned
    assert (desired in types.keys())

    if x.dtype == types[desired][-1]:
        print('Nothing to downcast: {} to {}'.format(x.dtype, desired))
        return x

    to_test = types[desired]
    if x.dtype in to_test:  # skip irrelevant downcasts
        to_test = types[desired][types[desired].index(x.dtype) + 1:]
    candidate = x
    for next_type in to_test:
        # if x.dtype == next_type:
        #    continue
        newcase = x.astype(next_type)
        if (newcase != x).sum() > 0:
            break
        candidate = newcase
    return candidate
