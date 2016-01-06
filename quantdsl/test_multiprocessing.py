import unittest

from multiprocessing import Array
from multiprocessing.pool import Pool

import os

import scipy

_d1 = None
_d2 = None

def init_process(*args):
    global _d1, _d2
    _d1 = args[0]
    _d2 = args[1]


def get_value(id):
    return numpy_from_array(v=_d1[id])


def set_value(id, value):
    _d2[id][:] = value


def numpy_from_array(v):
    # return scipy.frombuffer(v.get_obj())
    return scipy.frombuffer(v.get_obj())


def calc_result(id):
    value = get_value(id)
    assert value is not None
    result = value
    for i in range(1000):
        result = result * value
    set_value(id, result)


def echo(a):
    # for i in range(10000):
    #     a = a * a
    return a

class TestMultiprocessing(unittest.TestCase):

    def test_pickle_numpy(self):
        a = scipy.ones(200000)
        pool = Pool()
        results = pool.map(echo, [a] * 1000)
        for result in results:
            assert result.all() == scipy.ones(10000).all(), result

    def test_sharedmem(self):
        path_count = 2000
        d1 = {}
        d2 = {}
        node_count = 100
        for i in range(node_count):
            np1_vi = scipy.ones(path_count)
            v1 = Array('d', np1_vi)
            np2_vi = scipy.zeros(path_count)
            v2 = Array('d', np2_vi)
            d1[i] = v1
            d2[i] = v2
        p = Pool(processes=4, initializer=init_process, initargs=(d1, d2))
        results = []
        for i in range(node_count):
            results.append(p.apply_async(func=calc_result, args=(i,)))
        [r.get() for r in results]
        np_array = numpy_from_array(d2[0])
        np_array_mean = np_array.mean()
        self.assertEqual(np_array_mean, 1)
