import subprocess
import os
import numpy as np

BASE_DIR = os.path.dirname(os.path.realpath(__file__))

if subprocess.call(['make', '-C', BASE_DIR]) != 0:  # return value
    raise RuntimeError('Cannot compile pse: {}'.format(BASE_DIR))


def pse(polys, min_area):
    from .adaptor import pse as cpse
    # start = time.time()
    ret = np.array(cpse(polys, min_area), dtype='int32')
    # end = time.time()
    # print (end - start), 's'
    return ret
