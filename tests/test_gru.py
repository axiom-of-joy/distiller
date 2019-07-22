"""
This script tests the weight quantization sub-module for instances of
torch.nn.modules.GRU found in distiller/modules/gru.py.

Run with pytest -vv -s test_gru.py.

Author: Alexander Song
"""

from random import randrange, randint
import pytest
import numpy as np
import numpy.testing as nptest
import torch
import torch.nn as nn
from torch.nn.modules.rnn import GRU, GRUCell, LSTM, LSTMCell
import distiller
from distiller.modules.gru import (DistillerGRUCell,
                                   DistillerGRU,
                                   convert_model_to_distiller_gru)
from distiller.modules.rnn import (DistillerLSTMCell,
                                   DistillerLSTM,
                                   convert_model_to_distiller_lstm)


@pytest.mark.parametrize(
    "input_size,hidden_size,num_layers,sequence_len,batch_size",
    # Produces 4 random tuples of length 5.
    [tuple(randint(5, 20) for _ in range(5)) for _ in range(4)]
)
def test_distiller_gru_forward(input_size, hidden_size, num_layers,
                               sequence_len, batch_size):
    """
    Tests the forward methods of torch.nn.modules.GRU and DistillerGRU
    to ensure they produce the same output.
    """

    num_directions = 1

    torch_gru = GRU(input_size, hidden_size, num_layers)
    dist_gru = convert_model_to_distiller_gru(torch_gru)

    input_ = torch.randn(sequence_len, batch_size, input_size)
    h0 = torch.randn(num_layers * num_directions, batch_size, hidden_size)
    dist_output, dist_h = dist_gru(input_, h0)
    torch_output, torch_h = torch_gru(input_, h0)

    dist_output = dist_output.detach().numpy()
    torch_output = torch_output.detach().numpy()

    nptest.assert_array_almost_equal(torch_output, dist_output)


@pytest.mark.parametrize(
    "input_size,hidden_size,num_layers,sequence_len,batch_size",
    # Produces 4 random tuples of length 5.
    [tuple(randint(5, 20) for _ in range(5)) for _ in range(4)]
)
def test_distiller_lstm_forward(input_size, hidden_size, num_layers,
                                sequence_len, batch_size):
    """
    Tests the forward methods of torch.nn.modules.LSTM and DistillerLSTM
    to ensure they produce the same output.
    """

    num_directions = 1  # Uni-directional LSTM.
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


@pytest.mark.parametrize("input_size,hidden_size,batch_size", [
    (10, 20, 5),
    (1231, 3492, 39),
    (3492, 346, 349),
    (349, 423, 394),
])
def test_distiller_gru_cell_forward(input_size, hidden_size, batch_size):
    """
    Tests the forward methods of torch.nn.modules.GRUCell and
    DistillerGRUCell to ensure they produce the same output.
    """

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
    """
    Tests the forward methods of torch.nn.modules.LSTMCell and
    DistillerLSTMCell to ensure they produce the same output.
    """

    input_ = torch.randn(batch_size, input_size)
    hx = torch.randn(batch_size, hidden_size)
    dist_lstm_cell = DistillerLSTMCell(input_size, hidden_size)
    torch_lstm_cell = dist_lstm_cell.to_pytorch_impl()
    torch_output, _ = torch_lstm_cell(input_, (hx, hx))
    torch_output = torch_output.detach().numpy()
    dist_output, _ = dist_lstm_cell(input_, (hx, hx))
    dist_output = dist_output.detach().numpy()

    nptest.assert_array_almost_equal(torch_output, dist_output)
 
