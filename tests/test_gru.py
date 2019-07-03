# pytest -vv -s script_name.py

import torch
import torch.nn as nn
from torch.nn.modules.rnn import GRU, GRUCell, LSTM, LSTMCell
import distiller
from distiller.modules.gru import DistillerGRUCell, DistillerGRU
from distiller.modules.rnn import DistillerLSTMCell, DistillerLSTM
import numpy as np
import numpy.testing as nptest
import pytest
from random import randrange


@pytest.mark.parametrize("input_size,hidden_size,batch_size", [
    (10, 20, 5),
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


@pytest.mark.parametrize("input_size,hidden_size,batch_size", [
    (10, 20, 5),
    (1231, 3492, 39),
    (3492, 346, 349),
    (349, 423, 394),
])
def test_torch_lstm_cell_forward(input_size, hidden_size, batch_size):
    input_ = torch.randn(batch_size, input_size)
    hx = torch.randn(batch_size, hidden_size)
    dist_lstm_cell = DistillerLSTMCell(input_size, hidden_size)
    torch_lstm_cell = dist_lstm_cell.to_pytorch_impl()
    torch_output, _ = torch_lstm_cell(input_, (hx, hx))
    torch_output = torch_output.detach().numpy()
    dist_output, _ = dist_lstm_cell(input_, (hx, hx))
    dist_output = dist_output.detach().numpy()

    nptest.assert_array_almost_equal(torch_output, dist_output)
 

@pytest.mark.parametrize("input_size,hidden_size,batch_size", [
    (10, 20, 5),
    (1231, 3492, 39),
    (3492, 346, 349),
    (349, 423, 394),
])
def test_torch_lstm_gru_cell(input_size, hidden_size, batch_size):
    '''
    A test to make sure I am using the API correctly for both torch implementations.
    '''
    input_ = torch.randn(batch_size, input_size)
    hx = torch.randn(batch_size, hidden_size)
    gru_cell = GRUCell(input_size, hidden_size)
    lstm_cell = LSTMCell(input_size, hidden_size)
    gru_output = gru_cell(input_, hx)
    lstm_output, _ = lstm_cell(input_, (hx, hx))
    return gru_output.shape == lstm_output.shape


def _test_convert_to_torch_impl():
    input_size = 10
    hidden_size = 12
    batch_size = 3
    input_ = torch.randn(batch_size, input_size)
    dist_gru_cell = DistillerGRUCell(input_size, hidden_size)
    torch_gru_cell = dist_gru_cell.to_pytorch_impl()
    print(type(torch_gru_cell.weight_hh))
    

# Call these main scripts with python script_name.py
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


# Testing out the torch GRU implementation.
def main3():
    input_size = 10
    hidden_size = 20
    num_layers = 2
    sequence_len = 5
    batch_size = 3
    num_directions = 1
    torch_gru = GRU(input_size, hidden_size, num_layers)
    input_ = torch.randn(sequence_len, batch_size, input_size)
    h0 = torch.randn(num_layers * num_directions, batch_size, hidden_size)
    output, hn = torch_gru(input_, h0)
    print(output.shape)
    print(hn.shape)

# Testing out the Distiller GRU Implementation.
def main4():
    input_size = 10
    hidden_size = 20
    num_layers = 2
    sequence_len = 5
    batch_size = 3
    num_directions = 1
    dist_gru = DistillerGRU(input_size, hidden_size, num_layers)
    input_ = torch.randn(sequence_len, sequence_len, input_size)
    h0 = torch.randn(num_layers * num_directions, batch_size, hidden_size)
    output, hn = dist_gru(input_, h0)
    h, c = hn
    print(input_.shape)
    print(h0.shape)
    print(output.shape)
    print(h.shape)
    print(c.shape)

def main5():
    input_size = 10
    hidden_size = 20
    batch_size = 3
    input_ = torch.randn(batch_size, input_size)
    hx = torch.randn(batch_size, hidden_size)
    gru_cell = GRUCell(input_size, hidden_size)
    lstm_cell = LSTMCell(input_size, hidden_size)
    gru_output = gru_cell(input_, hx)
    lstm_output = lstm_cell(input_, (hx, hx))

     
if __name__ == "__main__":
    main4()
