# pytest -vv script_name.py

import torch
import torch.nn as nn
from torch.nn.modules.rnn import GRU, GRUCell
import distiller
from distiller.modules.gru import DistillerGRUCell, DistillerGRU
import numpy as np
import numpy.testing as nptest
import pytest
from random import randrange


def torch_gru_cell_forward(input_, hx):
    input_size = input_.shape[-1]
    hidden_size = hx.shape[-1]
    gru_cell = GRUCell(input_size, hidden_size)
    hx = gru_cell(input_, hx)
    return hx


def distiller_gru_cell_forward(input_, hx):
    input_size = input_.shape[-1]
    hidden_size = hx.shape[-1]
    gru_cell = DistillerGRUCell(input_size, hidden_size)
    hx, _ = gru_cell(input_, (hx, hx))
    return hx


@pytest.mark.parametrize("input_size,hidden_size,batch_size", [
    (1231, 3492, 39),
    (3492, 346, 349),
    (349, 423, 394),
])
def test_gru_cell_forward(input_size, hidden_size, batch_size):
    input_ = torch.randn(batch_size, input_size)
    hx = torch.randn(batch_size, hidden_size)
    dist_gru_cell = DistillerGRUCell(input_size, hidden_size)
    torch_gru_cell = dist_gru_cell.to_pytorch_impl()
    torch_output = torch_gru_cell(input_, hx).detach().numpy()
    dist_output, _ = dist_gru_cell(input_, (hx, hx))
    dist_output = dist_output.detach().numpy()

    nptest.assert_array_almost_equal(torch_output, dist_output)
 

def _test_convert_to_torch_impl():
    input_size = 10
    hidden_size = 12
    batch_size = 3
    input_ = torch.randn(batch_size, input_size)
    dist_gru_cell = DistillerGRUCell(input_size, hidden_size)
    torch_gru_cell = dist_gru_cell.to_pytorch_impl()
    print(type(torch_gru_cell.weight_hh))
    


def main():
    # Create inputs.
    input_size = 10
    hidden_size = 12
    batch_size = 3
    input_ = torch.randn(batch_size, input_size)
    hx = torch.randn(batch_size, hidden_size)
    torch_output = torch_gru_cell_forward(input_, hx)
    dist_output = distiller_gru_cell_forward(input_, hx)
    print(torch_output.shape)
    print(dist_output.shape)


def main2():
    # Create inputs.
    input_size = 10
    hidden_size = 12
    batch_size = 3
    torch_gru_cell = GRUCell(input_size, hidden_size)
    dist_gru_cell = DistillerGRUCell(input_size, hidden_size)
    torch_gru_cell = dist_gru_cell.to_pytorch_impl()
    print(type(dist_gru_cell))
    print(type(torch_gru_cell))
    #print(torch_gru_cell.weight_ih.shape)
    #torch_gru_cell.weight_hh
    #torch_gru_cell.bias_ih
    #torch_gru_cell.bias_hh

#    input_ = torch.randn(batch_size, input_size)
#    hx = torch.randn(batch_size, hidden_size)
#    torch_output = torch_gru_cell_forward(input_, hx)
#    dist_output = distiller_gru_cell_forward(input_, hx)
#    print(torch_output.shape)
#    print(dist_output.shape)


if __name__ == "__main__":
    main2()
