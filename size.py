import math


def size(size_bytes):
    if size_bytes < 1024:
        return '{} B'.format(size_bytes)
    size_name = ("B", "KiB", "MiB", "GiB", "TiB", "PiB", "EiB", "ZiB", "YiB")
    i = int(math.floor(math.log(size_bytes, 1024)))
    p = math.pow(1024, i)
    s = size_bytes / p  # s = round(size_bytes / p, 2)
    return '{:.1f} {}'.format(s, size_name[i])
