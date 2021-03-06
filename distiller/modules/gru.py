"""
A weight-quantization sub-module for instances of torch.nn.modules.GRU.

This sub-module is used for post-training weight quantization of GRUs.
It is adapted from the rnn.py sub-module for quantizing LSTMs. The main
changes are to the DistillerGRUCell class (adapted from the
DistillerLSTMCell class) to reflect the equations at
https://pytorch.org/docs/stable/nn.html#grucell. In addition,
DistillerGRU was adapted from DistillerLSTM to reflect the differences
between LSTMs and GRUs (i.e., the lack of a cell state in the case of
GRUs).

Tests for this script may be found in test/test_gru.py.

Author: Alexander Song
"""

#
# Copyright (c) 2019 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

import torch
import torch.nn as nn
import numpy as np
from .eltwise import EltwiseAdd, EltwiseMult
from itertools import product

__all__ = ['DistillerGRUCell', 'DistillerGRU', 'convert_model_to_distiller_gru']


class DistillerGRUCell(nn.Module):
    """
    A single GRU block.
    The calculation of the output takes into account the input and the previous output and cell state:
    https://pytorch.org/docs/stable/nn.html#grucell
    Args:
        input_size (int): the size of the input
        hidden_size (int): the size of the hidden state / output
        bias (bool): use bias. default: True

    """
    def __init__(self, input_size, hidden_size, bias=True):
        super(DistillerGRUCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias

        # Treat r, z, and n as one single object.
        self.fc_gate_x = nn.Linear(input_size, hidden_size * 3, bias=bias)
        self.fc_gate_h = nn.Linear(hidden_size, hidden_size * 3, bias=bias)
        self.eltwiseadd_gate = EltwiseAdd()
        self.eltwisemult_gate = EltwiseMult()
        # Apply activations separately:
        self.act_r = nn.Sigmoid()
        self.act_z = nn.Sigmoid()
        self.act_n = nn.Tanh()
        # Calculate hidden:
        self.init_weights()

    def forward(self, x, h=None):
        """
        Implemented as defined in https://pytorch.org/docs/stable/nn.html#grucell.
        """
        x_bsz, x_device = x.size(1), x.device
        if h is None:
            h = self.init_hidden(x_bsz, device=x_device)
        
        h_prev, _ = h
        fc_gate_x_, fc_gate_h_ = self.fc_gate_x(x), self.fc_gate_h(h_prev)
        r_x, z_x, n_x = torch.chunk(fc_gate_x_, 3, dim=1)
        r_h, z_h, n_h = torch.chunk(fc_gate_h_, 3, dim=1)
        r, z = self.eltwiseadd_gate(r_x, r_h), self.eltwiseadd_gate(z_x, z_h)
        r, z = self.act_r(r), self.act_z(z)
        n = self.eltwiseadd_gate(
            n_x,
            self.eltwisemult_gate(r, n_h)
        )
        n = self.act_n(n)

        # Construct h.
        minus_ones = torch.empty(z.shape, device=z.device)
        minus_ones.fill_(-1.0)
        minus_z = self.eltwisemult_gate(minus_ones, z)
        plus_ones = torch.ones(z.shape, device=minus_z.device)
        one_minus_z = self.eltwiseadd_gate(plus_ones, minus_z)
        h = self.eltwiseadd_gate(
            self.eltwisemult_gate(one_minus_z, n),
            self.eltwisemult_gate(z, h_prev)
        )
        return h, h

    def init_hidden(self, batch_size, device='cuda:0'):
        h_0 = torch.zeros(batch_size, self.hidden_size).to(device)
        return h_0, h_0

    def init_weights(self):
        initrange = 1 / np.sqrt(self.hidden_size)
        self.fc_gate_x.weight.data.uniform_(-initrange, initrange)
        self.fc_gate_h.weight.data.uniform_(-initrange, initrange)

    def to_pytorch_impl(self):
        module = nn.GRUCell(self.input_size, self.hidden_size, self.bias)
        module.weight_hh, module.weight_ih = \
            nn.Parameter(self.fc_gate_h.weight.clone().detach()), \
            nn.Parameter(self.fc_gate_x.weight.clone().detach())
        if self.bias:
            module.bias_hh, module.bias_ih = \
                nn.Parameter(self.fc_gate_h.bias.clone().detach()), \
                nn.Parameter(self.fc_gate_x.bias.clone().detach())
        return module

    @staticmethod
    def from_pytorch_impl(grucell: nn.GRUCell):
        module = DistillerGRUCell(input_size=grucell.input_size, hidden_size=grucell.hidden_size, bias=grucell.bias)
        module.fc_gate_x.weight = nn.Parameter(grucell.weight_ih.clone().detach())
        module.fc_gate_h.weight = nn.Parameter(grucell.weight_hh.clone().detach())
        if grucell.bias:
            module.fc_gate_x.bias = nn.Parameter(grucell.bias_ih.clone().detach())
            module.fc_gate_h.bias = nn.Parameter(grucell.bias_hh.clone().detach())

        return module

    def __repr__(self):
        return "%s(%d, %d)" % (self.__class__.__name__, self.input_size, self.hidden_size)


def process_sequence_wise(cell, x, h=None):
    """
    Process the entire sequence through an GRUCell.
    Args:
         cell (DistillerGRUCell): the cell.
         x (torch.Tensor): the input
         h (tuple of torch.Tensor-s): the hidden states of the GRUCell.
    Returns:
         y (torch.Tensor): the output
         h (tuple of torch.Tensor-s): the new hidden states of the GRUCell.
    """
    results = []
    for step in x:
        y, h = cell(step, h)
        results.append(y)
        h = (y, h)
    return torch.stack(results), h


def _repackage_hidden_unidirectional(h):
    """
    Repackages the hidden state into nn.GRU format. (unidirectional use)
    """
    h_all = [t[0] for t in h]
    c_all = [t[1] for t in h]
    return torch.stack(h_all, 0), torch.stack(c_all, 0)


def _repackage_hidden_bidirectional(h_result):
    """
    Repackages the hidden state into nn.GRU format. (bidirectional use)
    """
    h_all = [t[0] for t in h_result]
    c_all = [t[1] for t in h_result]
    return torch.cat(h_all, dim=0), torch.cat(c_all, dim=0)


def _unpack_bidirectional_input_h(h):
    """
    Unpack the bidirectional hidden states into states of the 2 separate directions.
    """
    h_t, c_t = h
    h_front, h_back = h_t[::2], h_t[1::2]
    c_front, c_back = c_t[::2], c_t[1::2]
    h_front = (h_front, c_front)
    h_back = (h_back, c_back)
    return h_front, h_back


class DistillerGRU(nn.Module):
    """
    A modular implementation of an GRU module.
    Args:
        input_size (int): size of the input
        hidden_size (int): size of the hidden connections and output.
        num_layers (int): number of GRUCells
        bias (bool): use bias
        batch_first (bool): the format of the sequence is (batch_size, seq_len, dim). default: False
        dropout : dropout factor
        bidirectional (bool): Whether or not the GRU is bidirectional. default: False (unidirectional).
        bidirectional_type (int): 1 or 2, corresponds to type 1 and type 2 as per
            https://github.com/pytorch/pytorch/issues/4930. default: 2
    """
    def __init__(self, input_size, hidden_size, num_layers, bias=True, batch_first=False,
                 dropout=0.5, bidirectional=False, bidirectional_type=2):
        super(DistillerGRU, self).__init__()
        if num_layers < 1:
            raise ValueError("Number of layers has to be at least 1.")
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.bias = bias
        self.batch_first = batch_first
        self.bidirectional_type = bidirectional_type

        if bidirectional:
            # Following https://github.com/pytorch/pytorch/issues/4930 -
            if bidirectional_type == 1:
                raise NotImplementedError
                # # Process each timestep at the entire layers chain -
                # # each timestep is forwarded through `front` and `back` chains independently,
                # # similarily to a unidirectional GRU.
                # self.cells = nn.ModuleList([GRUCell(input_size, hidden_size, bias)] +
                #                            [GRUCell(hidden_size, hidden_size, bias)
                #                             for _ in range(1, num_layers)])
                #
                # self.cells_reverse = nn.ModuleList([GRUCell(input_size, hidden_size, bias)] +
                #                                    [GRUCell(hidden_size, hidden_size, bias)
                #                                     for _ in range(1, num_layers)])
                # self.forward_fn = self.process_layer_wise
                # self.layer_chain_fn = self._layer_chain_bidirectional_type1

            elif bidirectional_type == 2:
                # Process the entire sequence at each layer consecutively -
                # the output of one layer is the sequence processed through the `front` and `back` cells
                # and the input to the next layers are both `output_front` and `output_back`.
                self.cells = nn.ModuleList([DistillerGRUCell(input_size, hidden_size, bias)] +
                                           [DistillerGRUCell(2 * hidden_size, hidden_size, bias)
                                            for _ in range(1, num_layers)])

                self.cells_reverse = nn.ModuleList([DistillerGRUCell(input_size, hidden_size, bias)] +
                                                   [DistillerGRUCell(2 * hidden_size, hidden_size, bias)
                                                    for _ in range(1, num_layers)])
                self.forward_fn = self._bidirectional_type2_forward

            else:
                raise ValueError("The only allowed types are [1, 2].")
        else:
            self.cells = nn.ModuleList([DistillerGRUCell(input_size, hidden_size, bias)] +
                                       [DistillerGRUCell(hidden_size, hidden_size, bias)
                                        for _ in range(1, num_layers)])
            self.forward_fn = self.process_layer_wise
            self.layer_chain_fn = self._layer_chain_unidirectional

        self.dropout = nn.Dropout(dropout)
        self.dropout_factor = dropout

    # A new forward function to match the torch GRU signature.
    def forward(self, x, h=None):
        y, hc = self._forward(x, (h, h) if h is not None else None)
        h, _ = hc
        return y, h

    # The original forward function with the LSTM-like function signature.
    def _forward(self, x, h=None):
        is_packed_seq = isinstance(x, nn.utils.rnn.PackedSequence)
        if is_packed_seq:
            return self.packed_sequence_forward(x, h)

        if self.batch_first:
            # Transpose to sequence_first format
            x = x.transpose(0, 1)
        x_bsz = x.size(1)

        if h is None:
            h = self.init_hidden(x_bsz)

        y, h = self.forward_fn(x, h)

        if self.batch_first:
            # Transpose back to batch_first format
            y = y.transpose(0, 1)
        return y, h

    def packed_sequence_forward(self, x, h=None):
        # Packed sequence treatment -
        # the sequences are not of the same size, hence
        # we split the padded tensor into the sequences.
        # we take the sequence from each row in the batch.
        x, lengths = nn.utils.rnn.pad_packed_sequence(x, batch_first=True)
        x_bsz = x.size(0)
        if h is None:
            h = self.init_hidden(x_bsz)
        y_results = []
        h_results = []
        for i, (sequence, seq_len) in enumerate(zip(x, lengths)):
            # Take the previous state according to the current batch.
            # we unsqueeze to have a 3D tensor
            h_current = (h[0][:, i, :].unsqueeze(1), h[1][:, i, :].unsqueeze(1))
            # Take only the relevant timesteps according to seq_len
            sequence = sequence[:seq_len].unsqueeze(1)  # sequence.shape = (seq_len, batch_size=1, input_dim)
            # forward pass:
            y, h_current = self.forward_fn(sequence, h_current)
            # sequeeze back the batch into a single sequence
            y_results.append(y.squeeze(1))
            h_results.append(h_current)
        # our result is a packed sequence
        y = nn.utils.rnn.pack_sequence(y_results)
        # concat hidden states per batches
        h = torch.cat([t[0] for t in h_results], dim=1), torch.cat([t[1] for t in h_results], dim=1)
        return y, h

    def process_layer_wise(self, x, h):
        results = []
        for step in x:
            y, h = self.layer_chain_fn(step, h)
            results.append(y)
        return torch.stack(results), h

    def _bidirectional_type2_forward(self, x, h):
        """
        Processes the entire sequence through a layer and passes the output sequence to the next layer.
        """
        out = x
        h_h_result = []
        h_c_result = []
        (h_front_all, c_front_all), (h_back_all, c_back_all) = _unpack_bidirectional_input_h(h)
        for i, (cell_front, cell_back) in enumerate(zip(self.cells, self.cells_reverse)):
            h_front, h_back = (h_front_all[i], c_front_all[i]), (h_back_all[i], c_back_all[i])

            # Sequence treatment:
            out_front, h_front = process_sequence_wise(cell_front, out, h_front)
            out_back, h_back = process_sequence_wise(cell_back, out.flip([0]), h_back)
            out = torch.cat([out_front, out_back.flip([0])], dim=-1)

            h_h_result += [h_front[0], h_back[0]]
            h_c_result += [h_front[1], h_back[1]]
            if i < self.num_layers-1:
                out = self.dropout(out)
        h = torch.stack(h_h_result, dim=0), torch.stack(h_c_result, dim=0)
        return out, h

    def _layer_chain_bidirectional_type1(self, x, h):
        # """
        # Process a single timestep through the entire bidirectional layer chain.
        # """
        # (h_front_all, c_front_all), (h_back_all, c_back_all) = _repackage_bidirectional_input_h(h)
        # h_result = []
        # out_front, out_back = x, x.flip([0])
        # for i, (cell_front, cell_back) in enumerate(zip(self.cells, self.cells_reverse)):
        #     h_front, h_back = (h_front_all[i], c_front_all[i]), (h_back_all[i], c_back_all[i])
        #     h_front, c_front = cell_front(out_front, h_front)
        #     h_back, c_back = cell_back(out_back, h_back)
        #     out_front, out_back = h_front, h_back
        #     if i < self.num_layers-1:
        #         out_front, out_back = self.dropout(out_front), self.dropout(out_back)
        #     h_current = torch.stack([h_front, h_back]), torch.stack([c_front, c_back])
        #     h_result.append(h_current)
        # h_result = _repackage_hidden_bidirectional(h_result)
        # return torch.cat([out_front, out_back], dim=-1), h_result
        raise NotImplementedError

    def _layer_chain_unidirectional(self, step, h):
        """
        Process a single timestep through the entire unidirectional layer chain.
        """
        step_bsz = step.size(0)
        if h is None:
            h = self.init_hidden(step_bsz)
        h_all, c_all = h
        h_result = []
        out = step
        for i, cell in enumerate(self.cells):
            h = h_all[i], c_all[i]
            out, hid = cell(out, h)
            if i < self.num_layers-1:
                out = self.dropout(out)
            h_result.append((out, hid))
        h_result = _repackage_hidden_unidirectional(h_result)
        return out, h_result

    def init_hidden(self, batch_size):
        weight = next(self.parameters())
        n_dir = 2 if self.bidirectional else 1
        return (weight.new_zeros(self.num_layers * n_dir, batch_size, self.hidden_size),
                weight.new_zeros(self.num_layers * n_dir, batch_size, self.hidden_size))

    def init_weights(self):
        for cell in self.hidden_cells:
            cell.init_weights()

    def flatten_parameters(self):
        pass

    def to_pytorch_impl(self):
        if self.bidirectional and self.bidirectional_type == 1:
            raise TypeError("Pytorch implementation of bidirectional GRU doesn't support type 1.")

        module = nn.GRU(input_size=self.input_size,
                         hidden_size=self.hidden_size,
                         num_layers=self.num_layers,
                         dropout=self.dropout_factor,
                         bias=self.bias,
                         batch_first=self.batch_first,
                         bidirectional=self.bidirectional)
        param_gates = ['i', 'h']

        param_types = ['weight']
        if self.bias:
            param_types.append('bias')

        suffixes = ['']
        if self.bidirectional:
            suffixes.append('_reverse')

        for i in range(self.num_layers):
            for ptype, pgate, psuffix in product(param_types, param_gates, suffixes):
                cell = self.cells[i] if psuffix == '' else self.cells_reverse[i]
                gru_pth_param_name = "%s_%sh_l%d%s" % (ptype, pgate, i, psuffix)  # e.g. `weight_ih_l0`
                gate_name = "fc_gate_%s" % ('x' if pgate == 'i' else 'h')  # `fc_gate_x` or `fc_gate_h`
                gate = getattr(cell, gate_name)  # e.g. `cell.fc_gate_x`
                param_tensor = getattr(gate, ptype).clone().detach()

                # same as `module.weight_ih_l0 = nn.Parameter(param_tensor)`:
                setattr(module, gru_pth_param_name, nn.Parameter(param_tensor))

        module.flatten_parameters()
        return module

    @staticmethod
    def from_pytorch_impl(gru: nn.GRU):
        bidirectional = gru.bidirectional

        module = DistillerGRU(gru.input_size, gru.hidden_size, gru.num_layers, bias=gru.bias,
                               batch_first=gru.batch_first,
                               dropout=gru.dropout, bidirectional=bidirectional)
        param_gates = ['i', 'h']

        param_types = ['weight']
        if gru.bias:
            param_types.append('bias')

        suffixes = ['']
        if bidirectional:
            suffixes.append('_reverse')

        for i in range(gru.num_layers):
            for ptype, pgate, psuffix in product(param_types, param_gates, suffixes):
                cell = module.cells[i] if psuffix == '' else module.cells_reverse[i]
                gru_pth_param_name = "%s_%sh_l%d%s" % (ptype, pgate, i, psuffix)  # e.g. `weight_ih_l0`
                gate_name = "fc_gate_%s" % ('x' if pgate == 'i' else 'h')  # `fc_gate_x` or `fc_gate_h`
                gate = getattr(cell, gate_name)  # e.g. `cell.fc_gate_x`
                param_tensor = getattr(gru, gru_pth_param_name).clone().detach()  # e.g. `gru.weight_ih_l0.detach()`
                setattr(gate, ptype, nn.Parameter(param_tensor))

        return module

    def __repr__(self):
        return "%s(%d, %d, num_layers=%d, dropout=%.2f, bidirectional=%s)" % \
               (self.__class__.__name__,
                self.input_size,
                self.hidden_size,
                self.num_layers,
                self.dropout_factor,
                self.bidirectional)


def convert_model_to_distiller_gru(model: nn.Module):
    """
    Replaces all `nn.GRU`s and `nn.GRUCell`s in the model with distiller versions.
    Args:
        model (nn.Module): the model
    """
    if isinstance(model, nn.GRUCell):
        return DistillerGRUCell.from_pytorch_impl(model)
    if isinstance(model, nn.GRU):
        return DistillerGRU.from_pytorch_impl(model)
    for name, module in model.named_children():
        module = convert_model_to_distiller_gru(module)
        setattr(model, name, module)

    return model
