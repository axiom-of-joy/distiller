# pytest -vv -s script_name.py

import torch
import torch.nn as nn
from torch.nn.modules.rnn import GRU, GRUCell, LSTM, LSTMCell
import distiller
from distiller.modules.gru import DistillerGRUCell, DistillerGRU, convert_model_to_distiller_gru
from distiller.modules.rnn import DistillerLSTMCell, DistillerLSTM, convert_model_to_distiller_lstm
import numpy as np
import numpy.testing as nptest
import pytest
from random import randrange, randint


@pytest.mark.parametrize("input_size,hidden_size,num_layers,sequence_len,batch_size",
    [tuple(randint(5, 20) for _ in range(5)) for _ in range(4)] # Produces 4 random tuples of length 5.
)
def test_distiller_lstm_forward(input_size, hidden_size, num_layers, sequence_len, batch_size):
    #input_size = 11
    #hidden_size = 7
    #num_layers = 2
    #sequence_len = 5
    #batch_size = 3
    num_directions = 1
    #dist_lstm = DistillerLSTM(input_size, hidden_size, num_layers)
    #torch_lstm = dist_lstm.to_pytorch_impl()
    torch_lstm = LSTM(input_size, hidden_size, num_layers)
    dist_lstm = convert_model_to_distiller_lstm(torch_lstm)
    
    # Display types.
    print(type(dist_lstm))
    print(type(torch_lstm))

    # Create input, hidden and cell states.
    torch.manual_seed(0)
    input_ = torch.randn(sequence_len, batch_size, input_size)
    h0 = torch.randn(num_layers * num_directions, batch_size, hidden_size)
    c0 = torch.randn(num_layers * num_directions, batch_size, hidden_size)

    # Calculate outputs.
    dist_output, dist_hn = dist_lstm(input_, (h0, c0))
    torch_output, torch_hn = torch_lstm(input_, (h0, c0))

    dist_output = dist_output.detach().numpy()
    torch_output = torch_output.detach().numpy()

    nptest.assert_array_almost_equal(torch_output, dist_output)

@pytest.mark.parametrize("input_size,hidden_size,num_layers,sequence_len,batch_size",
    [tuple(randint(5, 20) for _ in range(5)) for _ in range(4)] # Produces 4 random tuples of length 5.
)
def test_distiller_gru_forward(input_size, hidden_size, num_layers, sequence_len, batch_size):
    #input_size = 11
    #hidden_size = 7
    #num_layers = 2
    #sequence_len = 5
    #batch_size = 3
    num_directions = 1
    #dist_gru = DistillerGRU(input_size, hidden_size, num_layers)
    #torch_gru = dist_gru.to_pytorch_impl()

    torch_gru = GRU(input_size, hidden_size, num_layers)
    dist_gru = convert_model_to_distiller_gru(torch_gru)

    input_ = torch.randn(sequence_len, batch_size, input_size)
    h0 = torch.randn(num_layers * num_directions, batch_size, hidden_size)
    # These lines are for the old (LSTM-like) API.
    dist_output, dist_hn = dist_gru(input_, (h0, h0))
    dist_h, dist_c = dist_hn
    #dist_output, dist_h = dist_gru(input_, h0)
    torch_output, torch_h = torch_gru(input_, h0)

    dist_output = dist_output.detach().numpy()
    torch_output = torch_output.detach().numpy()

    nptest.assert_array_almost_equal(torch_output, dist_output)


@pytest.mark.parametrize("input_size,hidden_size,batch_size", [
    (10, 20, 5),
    (1231, 3492, 39),
    (3492, 346, 349),
    (349, 423, 394),
])
def test_distiller_gru_cell_forward(input_size, hidden_size, batch_size):
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
def test_distller_lstm_cell_forward(input_size, hidden_size, batch_size):
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
    torch_gru = dist_gru.to_pytorch_impl()
    torch.manual_seed(0)
    input_ = torch.randn(sequence_len, batch_size, input_size)
    h0 = torch.randn(num_layers * num_directions, batch_size, hidden_size)
    #FIXME
    import pudb
    pudb.set_trace()
    #
    dist_output, dist_hn = dist_gru(input_, (h0, h0))
    dist_h, dist_c = dist_hn

    torch_output, torch_hn = torch_gru(input_, h0)


    print(input_.shape)
    print(h0.shape)

    print(dist_output.shape)
    print(dist_h.shape)
    print(dist_c.shape)

    print(torch_output.shape)
    print(torch_hn.shape)


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

def main6():
    # Create inputs.
    input_size = 10
    hidden_size = 12
    batch_size = 3
    torch_gru_cell = GRUCell(input_size, hidden_size)
    #dist_gru_cell = DistillerGRUCell(input_size, hidden_size)
    #torch_gru_cell = dist_gru_cell.to_pytorch_impl()
    #print(type(dist_gru_cell))
    print(type(torch_gru_cell))
    input_ = torch.randn(batch_size, input_size)
    hidden = torch.randn(hidden_size)
    output = torch_gru_cell(input_, hidden)
    print(output.shape)


     
if __name__ == "__main__":
    main4()
