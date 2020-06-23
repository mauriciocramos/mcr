import os
import sys

# Reference: https://programtalk.com/python-examples/psutil.Process.memory_info_ex.peak_wset/

USING_WINDOWS = False
# Windows does not have a resource module
if sys.platform != 'win32':
    import resource
else:
    USING_WINDOWS = True
    import psutil


def get_memory_usage():
    """Cross platform method to get processor info"""
    if not USING_WINDOWS:
        # Some have said that this returns numbers in kb on some systems vs bytes on others.
        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    else:
        # Suggested from: http://stackoverflow.com/questions/9850995/tracking-maximum-memory-usage-by-a-python-function
        # this returns in bytes
        return psutil.Process(os.getpid()).memory_info_ex().peak_wset
