import torch
import torch.nn as nn
from torch.nn import Parameter

from collections import namedtuple
from typing import List, Tuple
from torch import Tensor

import typing
import torch.nn.functional as F
from torch.nn import init


LSTMState = namedtuple('LSTMState', ['hx', 'cx'])
BidirLSTMState = namedtuple('BidirLSTMState', ['forward_state', 'backward_state'])

def double_flatten_states(states):

    states = flatten_states([flatten_states(inner) for inner in states])
    return [hidden.view([-1] + list(hidden.shape[2:])) for hidden in states]

def flatten_states(states):
    states = list(zip(*states))
    assert len(states) == 2
    return [torch.stack(state) for state in states]


def init_stacked_lstm(num_layers, first_layer, other_layer, first_layer_args, other_layer_args):
    layers = [first_layer(*first_layer_args)] + [other_layer(*other_layer_args)
                                           for _ in range(num_layers - 1)]
    return nn.ModuleList(layers)


class StackedLSTM(torch.nn.Module):

    def __init__(self, num_layers, first_layer, other_layer, first_layer_args, other_layer_args):
        super(StackedLSTM, self).__init__()
        self.layers = init_stacked_lstm(num_layers, first_layer, other_layer, first_layer_args,
                                        other_layer_args)


    def __call__(self, *input_blah, **kwargs_blah) -> typing.Any:
        return super().__call__(*input_blah, **kwargs_blah)

    def forward(self, input, states, lengths=None):
        # type: (Tensor, List[Tuple[Tensor, Tensor]], Tensor) -> Tuple[Tensor, List[Tuple[Tensor, Tensor]]]
        # List[LSTMState]: One state per layer

        #input = [seq_len, batch, input_size]
        #states = [LSTMState([batch, hidden_size], [batch, hidden_size]) for _ in range(num_layers)]

        # output = [seq_len, batch, hidden_size]
        #output_states = [LSTMState([batch, hidden_size], [batch, hidden_size]) for _ in range(num_layers)]

        output_states = []

        output = input

        i = 0
        for rnn_layer in self.layers:
            state = states[i]
            output, out_state = rnn_layer(output, state, lengths=lengths)
            output_states += [out_state]
            i += 1
        return output, output_states


class LayerNorm(nn.Module):

    def __init__(self, features, eps=1e-12):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(features))
        self.beta = nn.Parameter(torch.zeros(features))
        self.eps = eps
        self.reset_parameters()


    def reset_parameters(self):
        pass

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        demeaned_squares = (x - mean) ** 2
        var = demeaned_squares.mean(-1, keepdim=True)
        var_filter = 0.01*(var < self.eps).int()

        normalized = (x - mean) / torch.sqrt(var + self.eps + var_filter)

        return self.gamma * normalized + self.beta


class LayerNormDropoutLSTMCell(torch.nn.Module):
    def __init__(self, input_size, hidden_size, forget_bias=1.0,
                 use_ff_dropout=True, ff_drop_prob=0.0, same_ff_mask=True,
                 use_recurrent_dropout=True, recurrent_drop_prob=0.0, same_recurrent_mask=True,
                 use_layer_norm=True):

        super(LayerNormDropoutLSTMCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.weight_ih = Parameter(torch.zeros(4 * hidden_size, input_size))
        self.weight_hh = Parameter(torch.zeros(4 * hidden_size, hidden_size))
        # The layernorms provide learnable biases
        self.forget_bias = forget_bias

        self.use_recurrent_dropout = use_recurrent_dropout
        self.recurrent_drop_prob = recurrent_drop_prob
        self.recurrent_dropout_mask = None
        self.same_recurrent_mask = same_recurrent_mask

        self.use_ff_dropout = use_ff_dropout
        self.ff_drop_prob = ff_drop_prob
        self.ff_dropout_mask = None
        self.same_ff_mask = same_ff_mask

        self.use_layer_norm = use_layer_norm

        ln = LayerNorm

        if not use_layer_norm:
            ln = nn.Identity

        #self.layernorm_i = ln(4 * hidden_size)
        #self.layernorm_h = ln(4 * hidden_size)
        self.layernorm_ingate = ln(hidden_size)
        self.layernorm_forgetgate = ln(hidden_size)
        self.layernorm_cellgate = ln(hidden_size)
        self.layernorm_outgate = ln(hidden_size)

        self.layernorm_c = ln(hidden_size)

        self.reset_parameters()

    def reset_parameters(self):

        input_size = self.weight_ih.shape[1]
        hidden_size = self.weight_hh.shape[1]

        four_times_hidden_size = self.weight_hh.shape[0]
        assert four_times_hidden_size == 4 * hidden_size

        gain = 0.5
        init.xavier_uniform_(self.weight_ih[:hidden_size], gain=gain)
        init.xavier_uniform_(self.weight_ih[hidden_size: (2 * hidden_size)], gain=gain)
        init.xavier_uniform_(self.weight_ih[(2 * hidden_size): (3 * hidden_size)], gain=gain)
        init.xavier_uniform_(self.weight_ih[(3 * hidden_size):], gain=gain)

        init.orthogonal_(self.weight_hh[:hidden_size])
        init.orthogonal_(self.weight_hh[hidden_size: (2 * hidden_size)])
        init.orthogonal_(self.weight_hh[(2 * hidden_size): (3 * hidden_size)])
        init.orthogonal_(self.weight_hh[(3 * hidden_size):])

    def __call__(self, *input_blah, **kwargs_blah) -> typing.Any:
        return super().__call__(*input_blah, **kwargs_blah)

    def forward(self, input, state):
        # type: (Tensor, Tuple[Tensor, Tensor]) -> Tuple[Tensor, Tuple[Tensor, Tensor]]

        # input = [batch, input_size]
        # state = LSTMState([batch, hidden_size], [batch, hidden_size])

        #hy_dropped = [batch, hidden_size]
        #(hy, cy) = LSTMState([batch, hidden_size], [batch, hidden_size])

        hx, cx = state

        igates = torch.mm(input, self.weight_ih.t())
        hgates = torch.mm(hx, self.weight_hh.t())

        gates = igates + hgates
        ingate, forgetgate, cellgate, outgate = gates.chunk(4, 1)

        if self.use_layer_norm:
            ingate = self.layernorm_ingate(ingate)
            forgetgate = self.layernorm_forgetgate(forgetgate)
            cellgate = self.layernorm_cellgate(cellgate)
            outgate = self.layernorm_outgate(outgate)


        ingate = torch.sigmoid(ingate)
        forgetgate = torch.sigmoid(forgetgate + self.forget_bias)
        cellgate = torch.tanh(cellgate)
        outgate = torch.sigmoid(outgate)

        if self.use_recurrent_dropout:
            if (not self.same_recurrent_mask) or (self.recurrent_dropout_mask is None):
                self.recurrent_dropout_mask = torch.ones_like(cellgate)
                self.recurrent_dropout_mask = F.dropout(self.recurrent_dropout_mask,
                                                        self.recurrent_drop_prob, self.training)

            cellgate = cellgate * self.recurrent_dropout_mask

        cy = (forgetgate * cx) + (ingate * cellgate)

        if self.use_layer_norm:
            cy = self.layernorm_c(cy)

        hy = outgate * torch.tanh(cy)
        hy_dropped = hy

        if self.use_ff_dropout:
            if (not self.same_ff_mask) or (self.ff_dropout_mask is None):
                self.ff_dropout_mask = torch.ones_like(hy)
                self.ff_dropout_mask = F.dropout(self.ff_dropout_mask,
                                                 self.ff_drop_prob, self.training)

            hy_dropped = hy * self.ff_dropout_mask

        return hy_dropped, (hy, cy)


class ResidualLSTMLayer(torch.nn.Module):
    def __init__(self, cell, *cell_args, **cell_kwargs):
        super(ResidualLSTMLayer, self).__init__()
        self.cell = cell(*cell_args, **cell_kwargs)  # type: LayerNormDropoutLSTMCell


    def __call__(self, *input_blah, **kwargs_blah) -> typing.Any:
        return super().__call__(*input_blah, **kwargs_blah)


    def forward(self, input, state, reset_dropout_masks=True):
        # type: (Tensor, Tuple[Tensor, Tensor], bool) -> Tuple[Tensor, Tuple[Tensor, Tensor]]

        # input = [seq_len, batch, hidden_size]
        # state = LSTMState([batch, hidden_size], [batch, hidden_size])

        # torch.stack(outputs) = [seq_len, batch, hidden_size]
        # state = LSTMState([batch, hidden_size], [batch, hidden_size])

        if reset_dropout_masks:
            self.cell.ff_dropout_mask = None
            self.cell.recurrent_dropout_mask = None

        inputs = input.unbind(0)
        outputs = []
        for i in range(len(inputs)):
            out, state = self.cell(inputs[i], state)
            outputs += [out+inputs[i]]

        state = LSTMState(state[0], state[1])

        return torch.stack(outputs), state


class ReverseResidualLSTMLayer(torch.nn.Module):
    def __init__(self, cell, *cell_args, **cell_kwargs):
        super(ReverseResidualLSTMLayer, self).__init__()
        self.cell = cell(*cell_args, **cell_kwargs)  # type: LayerNormDropoutLSTMCell


    def __call__(self, *input_blah, **kwargs_blah) -> typing.Any:
        return super().__call__(*input_blah, **kwargs_blah)


    def forward(self, input, state, reset_dropout_masks=True):
        # type: (Tensor, Tuple[Tensor, Tensor], bool) -> Tuple[Tensor, Tuple[Tensor, Tensor]]

        # input = [seq_len, batch, hidden_size]
        # state = LSTMState([batch, hidden_size], [batch, hidden_size])

        # torch.stack(reverse(outputs)) = [seq_len, batch, hidden_size]
        # state = LSTMState([batch, hidden_size], [batch, hidden_size])

        if reset_dropout_masks:
            self.cell.ff_dropout_mask = None
            self.cell.recurrent_dropout_mask = None

        inputs = reverse(input.unbind(0))
        outputs = []
        for i in range(len(inputs)):
            out, state = self.cell(inputs[i], state)
            outputs += [out+inputs[i]]

        state = LSTMState(state[0], state[1])

        return torch.stack(reverse(outputs)), state


class LSTMLayer(torch.nn.Module):
    def __init__(self, cell, *cell_args, **cell_kwargs):
        super(LSTMLayer, self).__init__()
        self.cell = cell(*cell_args, **cell_kwargs)  # type: LayerNormDropoutLSTMCell


    def __call__(self, *input_blah, **kwargs_blah) -> typing.Any:
        return super().__call__(*input_blah, **kwargs_blah)


    def forward(self, input, state, reset_dropout_masks=True, lengths=None):
        # type: (Tensor, Tuple[Tensor, Tensor], bool, Tensor) -> Tuple[Tensor, Tuple[Tensor, Tensor]]

        # input = [seq_len, batch, input_size]
        # state = LSTMState([batch, hidden_size], [batch, hidden_size])

        # torch.stack(outputs) = [seq_len, batch, hidden_size]
        # state = LSTMState([batch, hidden_size], [batch, hidden_size])

        if reset_dropout_masks:
            self.cell.ff_dropout_mask = None
            self.cell.recurrent_dropout_mask = None

        inputs = input.unbind(0)
        outputs = []
        for i in range(len(inputs)):

            out, state = self.cell(inputs[i], state)
            outputs += [out]

        state = LSTMState(state[0], state[1])
        return torch.stack(outputs), state


def reverse(lst):
    # type: (List[Tensor]) -> List[Tensor]
    return lst[::-1]


def flipBatch(data, lengths):

    #data = [seq_len, batch, input_size]

    assert data.shape[1] == len(lengths), "Dimension Mismatch!"
    data_reverse = data.clone()

    for i in range(data.shape[1]):

        length = int(lengths[i].item())

        if length > 0:
            data_reverse[:length, i, :] = data[:length, i, :].flip(dims=[0])

    return data_reverse


def create_mask(data, lengths):

    # data = [seq_len, batch, input_size]

    assert data.shape[1] == len(lengths), "Dimension Mismatch!"
    mask = torch.zeros_like(data)

    for i in range(data.shape[1]):

        length = int(lengths[i].item())

        if length > 0:
            mask[:length, i, :] = 1

    return mask


#state returned by ReverseLSTMLayer is None if lengths (because of padding)
class ReverseLSTMLayer(torch.nn.Module):
    def __init__(self, cell, *cell_args, **cell_kwargs):
        super(ReverseLSTMLayer, self).__init__()
        self.cell = cell(*cell_args, **cell_kwargs)  # type: LayerNormDropoutLSTMCell

    def __call__(self, *input_blah, **kwargs_blah) -> typing.Any:
        return super().__call__(*input_blah, **kwargs_blah)

    def forward(self, input, state, reset_dropout_masks=True, lengths=None):
        # type: (Tensor, Tuple[Tensor, Tensor], bool, Tensor) -> Tuple[Tensor, Tuple[Tensor, Tensor]]

        # input = [seq_len, batch, input_size]
        # state = LSTMState([batch, hidden_size], [batch, hidden_size])

        # torch.stack(reverse(outputs)) = [seq_len, batch, hidden_size]
        # state = LSTMState([batch, hidden_size], [batch, hidden_size])

        if reset_dropout_masks:
            self.cell.ff_dropout_mask = None
            self.cell.recurrent_dropout_mask = None

        if lengths is not None and lengths.numel():
            #seq_len = input.shape[0]
            #batch_size = input.shape[1]
            #input_size = input.shape[2]
            #input_flip = torch.zeros(seq_len, batch_size, input_size)
            input_flip = flipBatch(input, lengths)
            inputs = input_flip.unbind(0)
        else:
            inputs = reverse(input.unbind(0))

        outputs = []
        for i in range(len(inputs)):
            out, state = self.cell(inputs[i], state)
            outputs += [out]

        state = LSTMState(state[0], state[1])

        if lengths is not None and lengths.numel():
            outputs = torch.stack(outputs)
            outputs = flipBatch(outputs, lengths)
            state = None
        else:
            outputs = torch.stack(reverse(outputs))

        return outputs, state

#bidirLSTM_states returned by BidirLSTMLayer is None if lengths (because of padding)
class BidirLSTMLayer(torch.nn.Module):

    def __init__(self, cell, *cell_args, **cell_kwargs):
        super(BidirLSTMLayer, self).__init__()

        self.forward_layer = LSTMLayer(cell, *cell_args,  **cell_kwargs)
        self.backward_layer = ReverseLSTMLayer(cell, *cell_args,  **cell_kwargs)

    def __call__(self, *input_blah, **kwargs_blah) -> typing.Any:
        return super().__call__(*input_blah, **kwargs_blah)

    def forward(self, input, states, reset_dropout_masks=True, lengths=None):
        # type: (Tensor, List[Tuple[Tensor, Tensor]], bool, Tensor) -> Tuple[Tensor, BidirLSTMState]

        ## type: (Tensor, List[Tuple[Tensor, Tensor]]) -> Tuple[Tensor, List[Tuple[Tensor, Tensor]]]
        # List[LSTMState]: [forward LSTMState, backward LSTMState]

        # input = [seq_len, batch, input_size]
        # states = [LSTMState([batch, hidden_size], [batch, hidden_size]) for _ in range(2)]

        # torch.cat(outputs, -1) = [seq_len, batch, 2 * hidden_size]
        # bidirLSTM_states.forward_state = LSTMState([batch, hidden_size], [batch, hidden_size])
        # bidirLSTM_states.backward_state = LSTMState([batch, hidden_size], [batch, hidden_size])

        outputs = []
        output_states = []

        forward_state = states[0]
        out, out_state = self.forward_layer(input, forward_state, reset_dropout_masks)
        outputs += [out]
        output_states += [out_state]

        backward_state = states[1]
        out, out_state = self.backward_layer(input, backward_state, reset_dropout_masks, lengths=lengths)
        outputs += [out]
        output_states += [out_state]

        bidirLSTM_states = BidirLSTMState(output_states[0], output_states[1])

        if lengths is not None and lengths.numel():
            bidirLSTM_states = None

        return torch.cat(outputs, -1), bidirLSTM_states


class BidirResidualLSTMLayer(torch.nn.Module):

    def __init__(self, cell, *cell_args, **cell_kwargs):
        super(BidirResidualLSTMLayer, self).__init__()

        self.forward_layer = LSTMLayer(cell, *cell_args,  **cell_kwargs)
        self.backward_layer = ReverseLSTMLayer(cell, *cell_args,  **cell_kwargs)

    def __call__(self, *input_blah, **kwargs_blah) -> typing.Any:
        return super().__call__(*input_blah, **kwargs_blah)

    def forward(self, input, states, reset_dropout_masks=True):
        # type: (Tensor, List[Tuple[Tensor, Tensor]], bool) -> Tuple[Tensor, BidirLSTMState]

        ## type: (Tensor, List[Tuple[Tensor, Tensor]]) -> Tuple[Tensor, List[Tuple[Tensor, Tensor]]]
        # List[LSTMState]: [forward LSTMState, backward LSTMState]

        # input = [seq_len, batch, 2 * hidden_size]
        # states = [LSTMState([batch, hidden_size]), [batch, hidden_size]) for _ in range(2)]

        # outputs = [seq_len, batch, 2 * hidden_size]
        # bidirLSTM_states.forward_state = LSTMState([batch, hidden_size], [batch, hidden_size])
        # bidirLSTM_states.backward_state = LSTMState([batch, hidden_size], [batch, hidden_size])

        outputs = []
        output_states = []

        forward_state = states[0]
        out, out_state = self.forward_layer(input, forward_state, reset_dropout_masks)
        outputs += [out]
        output_states += [out_state]

        backward_state = states[1]
        out, out_state = self.backward_layer(input, backward_state, reset_dropout_masks)
        outputs += [out]
        output_states += [out_state]

        bidirLSTM_states = BidirLSTMState(output_states[0], output_states[1])

        outputs = torch.cat(outputs, -1)
        outputs += input

        return outputs, bidirLSTM_states
