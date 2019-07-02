# pytest -vv script_name.py

import torch
import torch.nn as nn
from torch.nn.modules.rnn import GRU, GRUCell
import distiller
from distiller.modules.gru import DistillerGRUCell, DistillerGRU
import numpy as np
import numpy.testing as nptest
import pytest


def run_torch_gru_cell(input_size, hidden_size, batch_size):
    gru_cell = GRUCell(input_size, hidden_size)
    input_ = torch.randn(batch_size, input_size)
    hx = torch.randn(batch_size, hidden_size)
    hx = gru_cell(input_, hx)
    return hx


def run_distiller_gru_cell(input_size, hidden_size, batch_size):
    gru_cell = DistillerGRUCell(input_size, hidden_size)
    input_ = torch.randn(batch_size, input_size)
    hx = torch.randn(batch_size, hidden_size)
    hx, _ = gru_cell(input_, (hx, hx))
    return hx

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


def main():
    input_size = 10
    hidden_size = 12
    batch_size = 3
    torch_output = run_torch_gru_cell(input_size, hidden_size, batch_size)
    dist_output = run_distiller_gru_cell(input_size, hidden_size, batch_size)
    print(torch_output.shape)
    print(dist_output.shape)


if __name__ == "__main__":
    main()
