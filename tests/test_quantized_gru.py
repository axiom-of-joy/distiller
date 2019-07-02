# pytest -vv script_name.py

import torch
import numpy as np
import numpy.testing as nptest
import pytest



def my_dec(func):

    def _wrapper(*args, **kwargs):
        for arg in args:
            res = func(*args, **kwargs)
        return res
    
    return _wrapper

def my_custom_dot(x, w):
    return x @ w


@pytest.mark.parametrize(
    'shapes', [
        ([1, 100], [100, 100]),
        ([100, 100], [100, 100]),
        ([1, 100], [100, 20]),
        ([1, 100, 20], [100, 20, 20]),
    ]
)
def test_dot_prod(shapes):

    x_shape, w_shape = shapes

    x = np.random.rand(*x_shape)
    w = np.random.rand(*w_shape)

    T = torch.mm(torch.tensor(x), torch.tensor(w))

    Y = my_custom_dot(x, w)

    nptest.assert_array_almost_equal(Y, T.numpy())

def test_thing():
    pass

def test_other_thing():
    pass
