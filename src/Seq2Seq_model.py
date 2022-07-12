import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

import typing

from lstm_models import StackedLSTM
from lstm_models import BidirLSTMLayer
from lstm_models import BidirResidualLSTMLayer
from lstm_models import LSTMLayer
from lstm_models import ResidualLSTMLayer
from lstm_models import LayerNormDropoutLSTMCell
from lstm_models import LSTMState


class MLP_network(nn.Module):

    def __init__(self, input_size, hidden_sizes):
        super().__init__()

        num_layers = len(hidden_sizes)
        layers = [nn.Linear(input_size, hidden_sizes[0])] + [nn.Linear(hidden_sizes[i], hidden_sizes[i+1])
                                                             for i in range(num_layers - 1)]

        self.layers = nn.ModuleList(layers)

    def __call__(self, *input, **kwargs) -> typing.Any:
        return super().__call__(*input, **kwargs)

    def forward(self, x):

        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x))

        return x


class Seq2Seq_VAE(nn.Module):

    def __init__(self, action_size, env_state_size, z_size, enc_hidden_size, dec_hidden_size,
                 num_layers, dec_num_layers=1, seq_len=10,
                 bidirectional_encoder=False, forget_bias=1.0, use_ff_dropout=False,
                 ff_drop_prob=0.0, same_ff_mask=True,
                 use_recurrent_dropout=True, recurrent_drop_prob=0.0, same_recurrent_mask=True,
                 use_layer_norm=True, num_hierarchies=2, mod_position=True):

        super().__init__()

        self.mod_position = mod_position

        if mod_position:
            policy_env_state_size = env_state_size - 1
        else:
            policy_env_state_size = env_state_size

        self.action_size = action_size
        self.env_state_size = env_state_size

        self.h1_z_size = z_size
        self.h2_z_size = z_size
        self.h3_z_size = z_size

        dec_hidden_size_h1 = dec_hidden_size
        dec_hidden_size_h2 = dec_hidden_size
        dec_hidden_size_h3 = dec_hidden_size

        self.h1_enc_hidden_size = enc_hidden_size
        self.h1_dec_hidden_size = dec_hidden_size_h1

        self.h2_bottom_enc_hidden_size = enc_hidden_size
        self.h2_top_enc_hidden_size = enc_hidden_size
        self.h2_dec_hidden_size = dec_hidden_size_h2

        self.h3_bottom_enc_hidden_size = enc_hidden_size
        self.h3_top_enc_hidden_size = enc_hidden_size
        self.h3_dec_hidden_size = dec_hidden_size_h3

        self.h1_num_layers = num_layers
        self.h1_dec_num_layers = dec_num_layers
        self.h2_num_layers = num_layers
        self.h2_dec_num_layers = dec_num_layers
        self.h3_num_layers = num_layers
        self.h3_dec_num_layers = dec_num_layers

        self.seq_len = seq_len

        self.h2_bottom_seq_len = self.seq_len
        self.h2_top_seq_len = seq_len

        self.h3_bottom_seq_len = 5
        self.h3_top_seq_len = 5

        self.bidirectional_encoder = bidirectional_encoder

        #this network no longer used
        self.h1_fc_initial_decoder_state = nn.Linear(self.h1_z_size, 2 * self.h1_dec_hidden_size)

        self.h1_fc_actions = nn.Linear(self.h1_dec_hidden_size, self.action_size)

        if num_hierarchies > 1:
            self.h2_fc_actions = nn.Linear(self.h2_dec_hidden_size, self.h1_z_size)

        if num_hierarchies > 2:
            self.h3_fc_actions = nn.Linear(self.h3_dec_hidden_size, self.h2_z_size)

        #encoder lstm
        if bidirectional_encoder:

            h1_encoder_stack_type = StackedLSTM
            h1_encoder_first_layer_type = BidirLSTMLayer
            h1_encoder_other_layer_type = BidirResidualLSTMLayer
            self.h1_encoder_dirs = 2

            h2_encoder_stack_type = StackedLSTM
            h2_encoder_first_layer_type = BidirLSTMLayer
            h2_encoder_other_layer_type = BidirResidualLSTMLayer
            self.h2_encoder_dirs = 2

            h3_encoder_stack_type = StackedLSTM
            h3_encoder_first_layer_type = BidirLSTMLayer
            h3_encoder_other_layer_type = BidirResidualLSTMLayer
            self.h3_encoder_dirs = 2
        else:
            h1_encoder_stack_type = StackedLSTM
            h1_encoder_first_layer_type = LSTMLayer
            h1_encoder_other_layer_type = ResidualLSTMLayer
            self.h1_encoder_dirs = 1

            h2_encoder_stack_type = StackedLSTM
            h2_encoder_first_layer_type = LSTMLayer
            h2_encoder_other_layer_type = ResidualLSTMLayer
            self.h2_encoder_dirs = 1

            h3_encoder_stack_type = StackedLSTM
            h3_encoder_first_layer_type = LSTMLayer
            h3_encoder_other_layer_type = ResidualLSTMLayer
            self.h3_encoder_dirs = 1


        h1_encoder_first_layer_args = [LayerNormDropoutLSTMCell, action_size + policy_env_state_size, self.h1_enc_hidden_size,
                                       forget_bias, use_ff_dropout, ff_drop_prob, same_ff_mask,
                                       use_recurrent_dropout, recurrent_drop_prob, same_recurrent_mask,
                                       use_layer_norm]

        h1_encoder_other_layer_args = [LayerNormDropoutLSTMCell, self.h1_enc_hidden_size * self.h1_encoder_dirs,
                                       self.h1_enc_hidden_size,
                                       forget_bias, use_ff_dropout, ff_drop_prob, same_ff_mask,
                                       use_recurrent_dropout, recurrent_drop_prob, same_recurrent_mask,
                                       use_layer_norm]

        self.h1_encoder_lstm = h1_encoder_stack_type(self.h1_num_layers,
                                                     h1_encoder_first_layer_type, h1_encoder_other_layer_type,
                                                     first_layer_args=h1_encoder_first_layer_args,
                                                     other_layer_args=h1_encoder_other_layer_args)

        self.h1_encode_mu = nn.Linear(self.h1_enc_hidden_size * self.h1_encoder_dirs, self.h1_z_size)
        self.h1_encode_logvar = nn.Linear(self.h1_enc_hidden_size * self.h1_encoder_dirs, self.h1_z_size)

        if num_hierarchies > 1:

            h2_bottom_encoder_first_layer_args = [LayerNormDropoutLSTMCell, policy_env_state_size, self.h2_bottom_enc_hidden_size,
                                                  forget_bias, use_ff_dropout, ff_drop_prob, same_ff_mask,
                                                  use_recurrent_dropout, recurrent_drop_prob, same_recurrent_mask,
                                                  use_layer_norm]

            h2_bottom_encoder_other_layer_args = [LayerNormDropoutLSTMCell, self.h2_bottom_enc_hidden_size * self.h2_encoder_dirs,
                                                  self.h2_bottom_enc_hidden_size,
                                                  forget_bias, use_ff_dropout, ff_drop_prob, same_ff_mask,
                                                  use_recurrent_dropout, recurrent_drop_prob, same_recurrent_mask,
                                                  use_layer_norm]

            self.h2_bottom_encoder_lstm = h2_encoder_stack_type(self.h2_num_layers,
                                                             h2_encoder_first_layer_type, h2_encoder_other_layer_type,
                                                             first_layer_args=h2_bottom_encoder_first_layer_args,
                                                             other_layer_args=h2_bottom_encoder_other_layer_args)


            h2_top_encoder_first_layer_args = [LayerNormDropoutLSTMCell, self.h2_bottom_enc_hidden_size * self.h2_encoder_dirs,
                                               self.h2_top_enc_hidden_size,
                                               forget_bias, use_ff_dropout, ff_drop_prob, same_ff_mask,
                                               use_recurrent_dropout, recurrent_drop_prob, same_recurrent_mask,
                                               use_layer_norm]

            h2_top_encoder_other_layer_args = [LayerNormDropoutLSTMCell, self.h2_top_enc_hidden_size * self.h2_encoder_dirs,
                                               self.h2_top_enc_hidden_size,
                                               forget_bias, use_ff_dropout, ff_drop_prob, same_ff_mask,
                                               use_recurrent_dropout, recurrent_drop_prob, same_recurrent_mask,
                                               use_layer_norm]

            self.h2_top_encoder_lstm = h2_encoder_stack_type(self.h2_num_layers,
                                                         h2_encoder_first_layer_type, h2_encoder_other_layer_type,
                                                         first_layer_args=h2_top_encoder_first_layer_args,
                                                         other_layer_args=h2_top_encoder_other_layer_args)

            self.h2_encode_mu = nn.Linear(self.h2_top_enc_hidden_size * self.h2_encoder_dirs, self.h2_z_size)
            self.h2_encode_logvar = nn.Linear(self.h2_top_enc_hidden_size * self.h2_encoder_dirs, self.h2_z_size)

        if num_hierarchies > 2:

            h3_bottom_encoder_first_layer_args = [LayerNormDropoutLSTMCell, self.h2_bottom_enc_hidden_size * self.h2_encoder_dirs,
                                                  self.h3_bottom_enc_hidden_size,
                                                  forget_bias, use_ff_dropout, ff_drop_prob, same_ff_mask,
                                                  use_recurrent_dropout, recurrent_drop_prob, same_recurrent_mask,
                                                  use_layer_norm]

            h3_bottom_encoder_other_layer_args = [LayerNormDropoutLSTMCell, self.h3_bottom_enc_hidden_size * self.h3_encoder_dirs,
                                                  self.h3_bottom_enc_hidden_size,
                                                  forget_bias, use_ff_dropout, ff_drop_prob, same_ff_mask,
                                                  use_recurrent_dropout, recurrent_drop_prob, same_recurrent_mask,
                                                  use_layer_norm]

            self.h3_bottom_encoder_lstm = h3_encoder_stack_type(self.h3_num_layers,
                                                             h3_encoder_first_layer_type, h3_encoder_other_layer_type,
                                                             first_layer_args=h3_bottom_encoder_first_layer_args,
                                                             other_layer_args=h3_bottom_encoder_other_layer_args)


            h3_top_encoder_first_layer_args = [LayerNormDropoutLSTMCell, self.h3_bottom_enc_hidden_size * self.h3_encoder_dirs,
                                               self.h3_top_enc_hidden_size,
                                               forget_bias, use_ff_dropout, ff_drop_prob, same_ff_mask,
                                               use_recurrent_dropout, recurrent_drop_prob, same_recurrent_mask,
                                               use_layer_norm]

            h3_top_encoder_other_layer_args = [LayerNormDropoutLSTMCell, self.h3_top_enc_hidden_size * self.h3_encoder_dirs,
                                               self.h3_top_enc_hidden_size,
                                               forget_bias, use_ff_dropout, ff_drop_prob, same_ff_mask,
                                               use_recurrent_dropout, recurrent_drop_prob, same_recurrent_mask,
                                               use_layer_norm]

            self.h3_top_encoder_lstm = h3_encoder_stack_type(self.h3_num_layers,
                                                         h3_encoder_first_layer_type, h3_encoder_other_layer_type,
                                                         first_layer_args=h3_top_encoder_first_layer_args,
                                                         other_layer_args=h3_top_encoder_other_layer_args)

            self.h3_encode_mu = nn.Linear(self.h3_top_enc_hidden_size * self.h3_encoder_dirs, self.h3_z_size)
            self.h3_encode_logvar = nn.Linear(self.h3_top_enc_hidden_size * self.h3_encoder_dirs, self.h3_z_size)

        h1_dec_hidden_sizes = [self.h1_dec_hidden_size] * self.h1_dec_num_layers
        self.h1_decoder_network = MLP_network(policy_env_state_size + self.h1_z_size, h1_dec_hidden_sizes)

        if num_hierarchies > 1:
            h2_dec_hidden_sizes = [self.h2_dec_hidden_size] * self.h2_dec_num_layers
            self.h2_decoder_network = MLP_network(policy_env_state_size + self.h2_z_size, h2_dec_hidden_sizes)

        if num_hierarchies > 2:
            h3_dec_hidden_sizes = [self.h3_dec_hidden_size] * self.h3_dec_num_layers
            self.h3_decoder_network = MLP_network(policy_env_state_size + self.h3_z_size, h3_dec_hidden_sizes)


    def __call__(self, *input_blah, **kwargs_blah) -> typing.Any:
        return super().__call__(*input_blah, **kwargs_blah)


    def encoder(self, inp, states, lengths=None, hierarchy_level="h1"):

        if hierarchy_level == "h1":
            if self.mod_position:
                inp = inp[:, :, 1:]
            out, out_states = self.h1_encoder_lstm(inp, states, lengths=lengths)
        elif hierarchy_level == "h2_top":
            out, out_states = self.h2_top_encoder_lstm(inp, states, lengths=lengths)
            h_enc_hidden_size = self.h2_top_enc_hidden_size
        elif hierarchy_level == "h2_bottom":
            if self.mod_position:
                inp = inp[:, :, 1:]
            out, out_states = self.h2_bottom_encoder_lstm(inp, states, lengths=lengths)
            h_enc_hidden_size = self.h2_bottom_enc_hidden_size
        elif hierarchy_level == "h3_top":
            out, out_states = self.h3_top_encoder_lstm(inp, states, lengths=lengths)
            h_enc_hidden_size = self.h3_top_enc_hidden_size
        elif hierarchy_level == "h3_bottom":
            out, out_states = self.h3_bottom_encoder_lstm(inp, states, lengths=lengths)
            h_enc_hidden_size = self.h3_bottom_enc_hidden_size

        batch_size = out.shape[1]
        last_forward_states = []
        last_backward_states = []

        if self.bidirectional_encoder:
            #first -1 index means taking last layer
            #second index is forward vs. backward state
            #third 0 index means taking hidden state instead of cell

            if hierarchy_level != "h1":
                for i in range(batch_size):
                    length = max(int(lengths[i].item()), 0)
                    last_forward_states += [out[length - 1, i, :h_enc_hidden_size]]
                    last_backward_states += [out[0, i, h_enc_hidden_size:]]

                last_forward_state = torch.stack(last_forward_states, 0)
                last_backward_state = torch.stack(last_backward_states, 0)
                last_state = torch.cat([last_forward_state, last_backward_state], -1)

            else:
                last_forward_state = out_states[-1][0][0]
                last_backward_state = out_states[-1][1][0]

                last_state = torch.cat([last_forward_state, last_backward_state], -1)
        else:

            if hierarchy_level != "h1":
                for i in range(batch_size):
                    last_forward_states += [out[lengths[i]-1, i, :h_enc_hidden_size]]

                last_state = torch.cat(last_forward_states, 1)
            else:
                last_state = out_states[-1].hx

        if hierarchy_level == "h1":
            mu = self.h1_encode_mu(last_state)
            logvar = self.h1_encode_logvar(last_state)
        elif hierarchy_level == "h2_top":
            mu = self.h2_encode_mu(last_state)
            logvar = self.h2_encode_logvar(last_state)
        elif hierarchy_level == "h2_bottom":
            mu = last_state
            logvar = None

        elif hierarchy_level == "h3_top":
            mu = self.h3_encode_mu(last_state)
            logvar = self.h3_encode_logvar(last_state)
        elif hierarchy_level == "h3_bottom":
            mu = last_state
            logvar = None

        return mu, logvar


    def decoder(self, inp, hierarchy_level="h1"):

        if self.mod_position:
            inp = inp[:, :, 1:]

        inputs = inp.unbind(0)
        outputs = []
        for i in range(len(inputs)):

            if hierarchy_level == "h1":
                out = self.h1_decoder_network(inputs[i])
            elif hierarchy_level == "h2":
                out = self.h2_decoder_network(inputs[i])
            elif hierarchy_level == "h3":
                out = self.h3_decoder_network(inputs[i])
            outputs += [out]

        out = torch.stack(outputs)

        return out

    def reparameterize(self, mu, logvar):

        std = 1e-6 + torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def seq_to_z(self, input_env_states, input_actions, lengths=None, hierarchy_level="h1"):

        #input_actions = [seq_len, batch_size, input_action_size]
        #input_env_states = [seq_len, batch_size, input_env_states_size]

        if hierarchy_level == "h1":

            seq_len = input_actions.shape[0]
            batch_size = input_actions.shape[1]

            assert seq_len == self.seq_len

            encoder_input = torch.cat([input_env_states, input_actions], -1)

            if self.h1_encoder_dirs == 2:
                encoder_initial_states = [[LSTMState(torch.zeros(batch_size, self.h1_enc_hidden_size),
                                                     torch.zeros(batch_size, self.h1_enc_hidden_size))
                                           for _ in range(2)]
                                          for _ in range(self.h1_num_layers)]
            else:
                encoder_initial_states = [LSTMState(torch.zeros(batch_size, self.h1_enc_hidden_size),
                                                    torch.zeros(batch_size, self.h1_enc_hidden_size))
                                          for _ in range(self.h1_num_layers)]

            mu, logvar = self.encoder(encoder_input, encoder_initial_states, hierarchy_level=hierarchy_level)
            sigma = torch.exp(logvar / 2.0)
            eps = torch.randn(batch_size, self.h1_z_size)

            batch_z = mu + torch.mul(sigma, eps)

            return mu, logvar, batch_z

        else:

            batch_size = input_env_states.shape[1]
            env_state_size = input_env_states.shape[2]

            input_env_states_subsequences = input_env_states.clone()

            if hierarchy_level == "h2":
                input_env_states_subsequences = \
                    input_env_states_subsequences.reshape(self.h2_top_seq_len, self.h2_bottom_seq_len,
                                                          batch_size, env_state_size)

                mu_list = []

                for i in range(self.h2_top_seq_len):

                    if self.h2_encoder_dirs == 2:
                        encoder_initial_states = [[LSTMState(torch.zeros(batch_size, self.h2_bottom_enc_hidden_size),
                                                             torch.zeros(batch_size, self.h2_bottom_enc_hidden_size))
                                                   for _ in range(2)]
                                                  for _ in range(self.h2_num_layers)]
                    else:
                        encoder_initial_states = [LSTMState(torch.zeros(batch_size, self.h2_bottom_enc_hidden_size),
                                                            torch.zeros(batch_size, self.h2_bottom_enc_hidden_size))
                                                  for _ in range(self.h2_num_layers)]

                    h2_bottom_lengths = torch.zeros(len(lengths))

                    for j in range(len(h2_bottom_lengths)):

                        d = lengths[j] - i*self.h2_bottom_seq_len
                        if d >= self.h2_bottom_seq_len:
                            h2_bottom_lengths[j] = self.h2_bottom_seq_len
                        else:
                            h2_bottom_lengths[j] = d

                    input_env_states_subsequence = input_env_states_subsequences[i]
                    mu, _ = self.encoder(input_env_states_subsequence, encoder_initial_states,
                                         lengths=h2_bottom_lengths, hierarchy_level="h2_bottom")

                    mu_list += [mu]

                #input_mus = [self.h2_top_seq_len, batch_size, 2*self.h2_bottom_enc_hidden_size]
                input_mus = torch.stack(mu_list)
                h2_top_lengths = 1 + ((lengths - 1) // self.h2_bottom_seq_len)

                if self.h2_encoder_dirs == 2:
                    encoder_initial_states = [[LSTMState(torch.zeros(batch_size, self.h2_top_enc_hidden_size),
                                                         torch.zeros(batch_size, self.h2_top_enc_hidden_size))
                                               for _ in range(2)]
                                              for _ in range(self.h2_num_layers)]
                else:
                    encoder_initial_states = [LSTMState(torch.zeros(batch_size, self.h2_top_enc_hidden_size),
                                                        torch.zeros(batch_size, self.h2_top_enc_hidden_size))
                                              for _ in range(self.h2_num_layers)]

                mu, logvar = self.encoder(input_mus, encoder_initial_states, lengths=h2_top_lengths,
                                          hierarchy_level="h2_top")

                sigma = torch.exp(logvar / 2.0)  # sigma > 0. div 2.0 -> sqrt.
                eps = torch.randn(batch_size, self.h2_z_size)
                batch_z = mu + torch.mul(sigma, eps)

                return mu, logvar, batch_z

            elif hierarchy_level == "h3":

                input_env_states_subsequences = \
                    input_env_states_subsequences.reshape(self.h3_top_seq_len, self.h3_bottom_seq_len,
                                                          self.h2_bottom_seq_len, batch_size, env_state_size)

                mu_h3_list = []

                for i in range(self.h3_top_seq_len):

                    sub_lengths = torch.zeros(len(lengths))

                    for k in range(len(sub_lengths)):

                        d = lengths[k] - i * self.h3_bottom_seq_len * self.h2_bottom_seq_len
                        if d >= self.h3_bottom_seq_len * self.h2_bottom_seq_len:
                            sub_lengths[k] = self.h3_bottom_seq_len * self.h2_bottom_seq_len
                        else:
                            sub_lengths[k] = d

                    input_env_states_subsequence_h3 = input_env_states_subsequences[i]
                    mu_h2_list = []

                    for j in range(self.h3_bottom_seq_len):

                        if self.h2_encoder_dirs == 2:
                            encoder_initial_states = [
                                [LSTMState(torch.zeros(batch_size, self.h2_bottom_enc_hidden_size),
                                           torch.zeros(batch_size, self.h2_bottom_enc_hidden_size))
                                 for _ in range(2)]
                                for _ in range(self.h2_num_layers)]
                        else:
                            encoder_initial_states = [LSTMState(torch.zeros(batch_size, self.h2_bottom_enc_hidden_size),
                                                                torch.zeros(batch_size, self.h2_bottom_enc_hidden_size))
                                                      for _ in range(self.h2_num_layers)]

                        h2_bottom_lengths = torch.zeros(len(sub_lengths))

                        for k in range(len(h2_bottom_lengths)):

                            d = sub_lengths[k] - j * self.h2_bottom_seq_len
                            if d >= self.h2_bottom_seq_len:
                                h2_bottom_lengths[k] = self.h2_bottom_seq_len
                            else:
                                h2_bottom_lengths[k] = d

                        input_env_states_subsequence = input_env_states_subsequence_h3[j]
                        mu, _ = self.encoder(input_env_states_subsequence, encoder_initial_states,
                                             lengths=h2_bottom_lengths, hierarchy_level="h2_bottom")

                        mu_h2_list += [mu]

                    # input_mus_h2 = [self.h3_bottom_seq_len, batch_size, 2*self.h2_bottom_enc_hidden_size]
                    input_mus_h2 = torch.stack(mu_h2_list)

                    h3_bottom_lengths = 1 + ((sub_lengths - 1) // self.h2_bottom_seq_len)

                    if self.h3_encoder_dirs == 2:
                        encoder_initial_states = [[LSTMState(torch.zeros(batch_size, self.h3_bottom_enc_hidden_size),
                                                             torch.zeros(batch_size, self.h3_bottom_enc_hidden_size))
                                                   for _ in range(2)]
                                                  for _ in range(self.h3_num_layers)]
                    else:
                        encoder_initial_states = [LSTMState(torch.zeros(batch_size, self.h3_bottom_enc_hidden_size),
                                                            torch.zeros(batch_size, self.h3_bottom_enc_hidden_size))
                                                  for _ in range(self.h3_num_layers)]

                    mu, _ = self.encoder(input_mus_h2, encoder_initial_states, lengths=h3_bottom_lengths,
                                         hierarchy_level="h3_bottom")

                    mu_h3_list += [mu]

                # input_mus_h3 = [self.h3_top_seq_len, batch_size, self.h3_bottom_enc_hidden_size]
                input_mus_h3 = torch.stack(mu_h3_list)
                h3_top_lengths = 1 + ( (lengths - 1) // (self.h3_bottom_seq_len * self.h2_bottom_seq_len) )

                if self.h3_encoder_dirs == 2:
                    encoder_initial_states = [[LSTMState(torch.zeros(batch_size, self.h3_top_enc_hidden_size),
                                                         torch.zeros(batch_size, self.h3_top_enc_hidden_size))
                                               for _ in range(2)]
                                              for _ in range(self.h3_num_layers)]
                else:
                    encoder_initial_states = [LSTMState(torch.zeros(batch_size, self.h3_top_enc_hidden_size),
                                                        torch.zeros(batch_size, self.h3_top_enc_hidden_size))
                                              for _ in range(self.h3_num_layers)]

                mu, logvar = self.encoder(input_mus_h3, encoder_initial_states, lengths=h3_top_lengths,
                                          hierarchy_level="h3_top")

                sigma = torch.exp(logvar / 2.0)  # sigma > 0. div 2.0 -> sqrt.
                eps = torch.randn(batch_size, self.h3_z_size)
                batch_z = mu + torch.mul(sigma, eps)

                return mu, logvar, batch_z


    def z_and_input_to_seq(self, input_env_states, batch_z, hierarchy_level="h1"):

        # input_env_states = [seq_len, batch_size, input_env_states_size]
        # input_actions = [seq_len, batch_size, input_action_size]
        # batch_z = [batch_size, z_size]

        #output = [seq_len, batch_size, hidden_size]

        if hierarchy_level == "h1":

            seq_len = input_env_states.shape[0]
            batch_size = input_env_states.shape[1]

            decoder_input = input_env_states

            pre_tile_z = torch.reshape(batch_z, [1, batch_size, self.h1_z_size])
            tiled_z = pre_tile_z.repeat(seq_len, 1, 1)

            decoder_input_with_z = torch.cat([decoder_input, tiled_z], -1)
            output = self.decoder(decoder_input_with_z, hierarchy_level=hierarchy_level)

            return output

        else:

            batch_size = input_env_states.shape[1]
            env_state_size = input_env_states.shape[2]

            input_env_states_subsequences = input_env_states.clone()

            if hierarchy_level == "h2":

                input_env_states_subsequences = \
                    input_env_states_subsequences.reshape(self.h2_top_seq_len, self.h2_bottom_seq_len,
                                                          batch_size, env_state_size)

                output_list = []

                for i in range(self.h2_top_seq_len):

                    input_env_states_subsequence = input_env_states_subsequences[i]
                    first_env_state = input_env_states_subsequence[0]
                    decoder_input_with_z = torch.cat([first_env_state, batch_z], -1)

                    output = self.decoder(decoder_input_with_z.unsqueeze(0), hierarchy_level="h2")

                    h1_z_raw = self.h2_fc_actions(output)
                    h1_z = 2.5 * torch.tanh(h1_z_raw)

                    output = self.z_and_input_to_seq(input_env_states_subsequence, h1_z,
                                                                 hierarchy_level="h1")

                    output_list += [output]

                output_list = torch.cat(output_list, dim=0)
                return output_list

            elif hierarchy_level == "h3":

                input_env_states_subsequences = \
                    input_env_states_subsequences.reshape(self.h3_top_seq_len, self.h3_bottom_seq_len,
                                                          self.h2_bottom_seq_len, batch_size, env_state_size)

                output_list = []

                for i in range(self.h3_top_seq_len):

                    input_env_states_subsequence_h3 = input_env_states_subsequences[i]
                    first_env_state = input_env_states_subsequence_h3[0, 0]
                    decoder_input_with_z = torch.cat([first_env_state, batch_z], -1)

                    output = self.decoder(decoder_input_with_z.unsqueeze(0), hierarchy_level="h3")

                    h2_z_raw = self.h3_fc_actions(output)
                    h2_z = 2.5 * torch.tanh(h2_z_raw)

                    for j in range(self.h3_bottom_seq_len):

                        input_env_states_subsequence = input_env_states_subsequence_h3[j]
                        first_env_state = input_env_states_subsequence[0]
                        decoder_input_with_z = torch.cat([first_env_state, h2_z.squeeze(0)], -1)

                        output = self.decoder(decoder_input_with_z.unsqueeze(0), hierarchy_level="h2")

                        h1_z_raw = self.h2_fc_actions(output)
                        h1_z = 2.5 * torch.tanh(h1_z_raw)

                        output = self.z_and_input_to_seq(input_env_states_subsequence, h1_z,
                                                                     hierarchy_level="h1")

                        output_list += [output]

                output_list = torch.cat(output_list, dim=0)
                return output_list


    def forward(self, input_env_states, input_actions, lengths=None, hierarchy_level="h1"):

        #input_actions = [seq_len, batch_size, input_action_size]

        mu, logvar, batch_z = self.seq_to_z(input_env_states, input_actions, lengths, hierarchy_level=hierarchy_level)
        output = self.z_and_input_to_seq(input_env_states, batch_z, hierarchy_level=hierarchy_level)

        output = torch.reshape(output, [-1, self.h1_dec_hidden_size])

        output_log_probs = F.log_softmax(self.h1_fc_actions(output), dim=-1)
        output_log_probs = output_log_probs.double()

        return output_log_probs, mu, logvar, batch_z


    def argmax_or_repulsion_primitive_from_region_bootstrap(self, pos_to_sample, z_grid, z_seq_len=10,
                                                  low_repulsion_idx=0, high_repulsion_idx=9,
                                                  argmax=True, env=None, bootstrap_level="h2", old_model=None):

        # pos_to_sample = [num_trajectories, z_size]
        # trajectory_actions = [num_trajectories, seq_len, 1]
        # trajectory_actions_one_hot = [num_trajectories, seq_len, action_size]
        # trajectory_probs = [num_trajectories, seq_len, action_size]
        # trajectory_states = [num_trajectories, seq_len, env_state_size]

        with torch.no_grad():

            num_trajectories = pos_to_sample.shape[0]

            trajectory_actions = []
            trajectory_actions_one_hot = []
            trajectory_probs = []
            trajectory_states = []

            trajectory_lengths_list = []

            fraction_h2 = 0.4

            for i in range(num_trajectories):

                trajectory_actions_single_seq = []
                trajectory_actions_one_hot_single_seq = []
                trajectory_probs_single_seq = []
                trajectory_states_single_seq = []

                initial_env_state = env.reset()
                initial_env_state = torch.from_numpy(initial_env_state).float()

                modify_trajectory_len = True
                trajectory_len = self.h2_bottom_seq_len

                num_env_states = 600
                length = 0
                trajectory_lengths = []

                while length < num_env_states:

                    random_index = torch.randint(num_trajectories, (1,))[0]
                    z_sample_long = z_grid[random_index].unsqueeze(0)

                    if i < fraction_h2 * num_trajectories:

                        output_actions, output_env_states, last_env_state, \
                        output_probs, h_zs, done, last_index = \
                            old_model.argmax_trajectory_h(z_sample_long, initial_env_state, env=env,
                                                          hierarchy_level="h2", random_traj=False,
                                                          modify_trajectory_len=modify_trajectory_len,
                                                          trajectory_len=trajectory_len)

                    else:

                        output_actions, output_env_states, last_env_state, \
                        output_probs, h_zs, done, last_index = \
                            old_model.argmax_trajectory_h(z_sample_long, initial_env_state, env=env,
                                                          hierarchy_level=bootstrap_level, random_traj=False,
                                                          modify_trajectory_len=modify_trajectory_len,
                                                          trajectory_len=trajectory_len)

                    length += last_index

                    if done and length < num_env_states:
                        trajectory_lengths += [length]

                    output_actions = output_actions[:last_index]
                    output_env_states = output_env_states[:last_index]
                    output_probs = output_probs[:last_index]

                    initial_env_state = env.reset()
                    initial_env_state = torch.from_numpy(initial_env_state).float()

                    classes = torch.tensor(range(self.action_size))
                    outputs_class = torch.sum(classes * output_actions, dim=-1).long()

                    trajectory_actions_single_seq += [outputs_class]
                    trajectory_actions_one_hot_single_seq += [output_actions]
                    trajectory_probs_single_seq += [output_probs]
                    trajectory_states_single_seq += [output_env_states]

                trajectory_actions_single_seq = torch.cat(trajectory_actions_single_seq, dim=0)
                trajectory_actions_one_hot_single_seq = torch.cat(trajectory_actions_one_hot_single_seq, dim=0)
                trajectory_probs_single_seq = torch.cat(trajectory_probs_single_seq, dim=0)
                trajectory_states_single_seq = torch.cat(trajectory_states_single_seq, dim=0)

                trajectory_actions_single_seq = trajectory_actions_single_seq[:num_env_states]
                trajectory_actions_one_hot_single_seq = trajectory_actions_one_hot_single_seq[:num_env_states]
                trajectory_probs_single_seq = trajectory_probs_single_seq[:num_env_states]
                trajectory_states_single_seq = trajectory_states_single_seq[:num_env_states]

                trajectory_actions += [trajectory_actions_single_seq]
                trajectory_actions_one_hot += [trajectory_actions_one_hot_single_seq]
                trajectory_probs += [trajectory_probs_single_seq]
                trajectory_states += [trajectory_states_single_seq]

                trajectory_lengths_list += [trajectory_lengths]

            trajectory_actions = torch.cat(trajectory_actions, dim=1)
            trajectory_actions_one_hot = torch.cat(trajectory_actions_one_hot, dim=1)
            trajectory_probs = torch.cat(trajectory_probs, dim=1)
            trajectory_states = torch.cat(trajectory_states, dim=1)

            trajectory_actions = trajectory_actions.unsqueeze(2)
            trajectory_actions = trajectory_actions.permute(1, 0, 2)
            trajectory_actions_one_hot = trajectory_actions_one_hot.permute(1, 0, 2)
            trajectory_probs = trajectory_probs.permute(1, 0, 2)
            trajectory_states = trajectory_states.permute(1, 0, 2)

            # trajectory_states = [num_trajectories, seq_len, env_state_size]

            for i in range(num_trajectories):

                z_sample = pos_to_sample[i].unsqueeze(0)

                for j in range(trajectory_actions.shape[1]):
                    initial_env_state = trajectory_states[i, j]

                    output_action, output_env_state, last_env_state, \
                    output_prob, done, t = \
                        self.argmax_primitive_seq(z_sample, initial_env_state=initial_env_state,
                                                  env=env, modify_trajectory_len=True, trajectory_len=1)

                    classes = torch.tensor(range(self.action_size))
                    output_class = torch.sum(classes * output_action).long()
                    trajectory_actions[i, j] = output_class
                    trajectory_actions_one_hot[i, j] = output_action
                    trajectory_probs[i, j] = output_prob

            z_seq_len_half = z_seq_len // 2
            x_indices = torch.arange(0, num_trajectories)
            x_indices = torch.stack([x_indices]*z_seq_len, dim=1)

            edge_indices_np = np.zeros((num_trajectories, z_seq_len_half), dtype=int)
            num_trajectory_edges = np.zeros(num_trajectories, dtype=int)

            for i in range(num_trajectories):

                trajectory_edge_indices = []

                for j in range(len(trajectory_lengths_list[i])):

                    length = trajectory_lengths_list[i][j]
                    trajectory_edge_indices += [max(length - 7, 0), max(length - 1, 0)]

                num_trajectory_edges[i] = len(trajectory_edge_indices)

                random_indices = np.random.randint(num_env_states, size=z_seq_len_half)
                trajectory_edge_indices_np = np.array(trajectory_edge_indices)
                trajectory_edge_indices_np = np.concatenate((trajectory_edge_indices_np, random_indices))
                trajectory_edge_indices_np = trajectory_edge_indices_np[:z_seq_len_half]

                edge_indices_np[i] = trajectory_edge_indices_np

            edge_indices = torch.from_numpy(edge_indices_np)

            selected_indices = torch.randint(num_env_states, (num_trajectories, z_seq_len_half))
            y_indices = torch.cat([edge_indices, selected_indices], dim=1)

            trajectory_actions_selected = trajectory_actions[x_indices, y_indices]
            trajectory_actions_one_hot_selected = trajectory_actions_one_hot[x_indices, y_indices]
            trajectory_probs_selected = trajectory_probs[x_indices, y_indices]
            trajectory_states_selected = trajectory_states[x_indices, y_indices]

            if not argmax:

                trajectory_probs_selected_np = trajectory_probs_selected.clone().numpy()
                max_probs = np.max(trajectory_probs_selected_np, axis=2)

                max_probs_argsorted = max_probs.argsort(axis=1)
                arg_maxes = np.argmax(trajectory_probs_selected_np, axis=2)

                uniform_probs_minus_argmax = (1.0 / (self.action_size - 1)) * \
                                                 np.ones((num_trajectories, z_seq_len, self.action_size - 1))

                action_indices = multinomial_sample(uniform_probs_minus_argmax, axis=2)
                action_indices[action_indices >= arg_maxes] += 1

                trajectory_actions_selected_np = trajectory_actions_selected.clone().numpy()

                for i in range(max_probs.shape[0]):

                    low_prob = max_probs[i, max_probs_argsorted[i, low_repulsion_idx]]
                    high_prob = max_probs[i, max_probs_argsorted[i, high_repulsion_idx]]

                    for j in range(z_seq_len):

                        if (j < num_trajectory_edges[i]) or \
                                ((max_probs[i, j] > low_prob - 0.001) and (max_probs[i, j] < high_prob + 0.001)):

                            trajectory_actions_selected_np[i, j] = action_indices[i, j]

                trajectory_actions_selected = torch.from_numpy(trajectory_actions_selected_np)
                trajectory_actions_one_hot_selected = torch.zeros(trajectory_actions_one_hot_selected.shape)
                trajectory_actions_one_hot_selected.scatter_(-1, trajectory_actions_selected, 1)

            return trajectory_actions_selected, trajectory_actions_one_hot_selected, \
                   trajectory_probs_selected, trajectory_states_selected


    def argmax_or_repulsion_primitive_from_region(self, pos_to_sample, z_seq_len=10, primitive_trajectory_len=5,
                                                  low_repulsion_idx=0, high_repulsion_idx=9,
                                                  argmax=True, env=None):

        # pos_to_sample = [num_trajectories, z_size]
        # trajectory_actions = [num_trajectories, seq_len, 1]
        # trajectory_actions_one_hot = [num_trajectories, seq_len, action_size]
        # trajectory_probs = [num_trajectories, seq_len, action_size]
        # trajectory_states = [num_trajectories, seq_len, env_state_size]

        with torch.no_grad():

            num_trajectories = pos_to_sample.shape[0]

            random_indices = torch.randint(num_trajectories, (num_trajectories, z_seq_len))
            long_trajectories_filler_z = pos_to_sample[random_indices]

            trajectory_actions = []
            trajectory_actions_one_hot = []
            trajectory_probs = []
            trajectory_states = []

            for i in range(num_trajectories):

                trajectory_actions_single_seq = []
                trajectory_actions_one_hot_single_seq = []
                trajectory_probs_single_seq = []
                trajectory_states_single_seq = []

                initial_env_state = env.reset()
                initial_env_state = torch.from_numpy(initial_env_state).float()

                for j in range(z_seq_len):

                    z_sample = pos_to_sample[i].unsqueeze(0)
                    z_sample_filler = long_trajectories_filler_z[i, j].unsqueeze(0)

                    output_actions, output_env_states, last_env_state, \
                        output_probs, done, t = \
                        self.argmax_primitive_seq(z_sample, initial_env_state=initial_env_state,
                                                  env=env, modify_trajectory_len=True,
                                                  trajectory_len=primitive_trajectory_len)

                    initial_env_state_filler = last_env_state
                    initial_env_state_filler = initial_env_state_filler.float()

                    output_actions_filler, output_env_states_filler, last_env_state_filler, \
                        output_probs_filler, done, t = \
                        self.argmax_primitive_seq(z_sample_filler, initial_env_state=initial_env_state_filler,
                                                  env=env, modify_trajectory_len=True, trajectory_len=primitive_trajectory_len)

                    initial_env_state = last_env_state_filler
                    initial_env_state = initial_env_state.float()

                    classes = torch.tensor(range(self.action_size))
                    outputs_class = torch.sum(classes * output_actions, dim=-1).long()
                    outputs_class_filler = torch.sum(classes * output_actions_filler, dim=-1).long()

                    trajectory_actions_single_seq += [outputs_class]
                    trajectory_actions_one_hot_single_seq += [output_actions]
                    trajectory_probs_single_seq += [output_probs]
                    trajectory_states_single_seq += [output_env_states]

                    trajectory_actions_single_seq += [outputs_class_filler]
                    trajectory_actions_one_hot_single_seq += [output_actions_filler]
                    trajectory_probs_single_seq += [output_probs_filler]
                    trajectory_states_single_seq += [output_env_states_filler]

                trajectory_actions_single_seq = torch.cat(trajectory_actions_single_seq, dim=0)
                trajectory_actions_one_hot_single_seq = torch.cat(trajectory_actions_one_hot_single_seq, dim=0)
                trajectory_probs_single_seq = torch.cat(trajectory_probs_single_seq, dim=0)
                trajectory_states_single_seq = torch.cat(trajectory_states_single_seq, dim=0)

                trajectory_actions += [trajectory_actions_single_seq]
                trajectory_actions_one_hot += [trajectory_actions_one_hot_single_seq]
                trajectory_probs += [trajectory_probs_single_seq]
                trajectory_states += [trajectory_states_single_seq]

            trajectory_actions = torch.cat(trajectory_actions, dim=1)
            trajectory_actions_one_hot = torch.cat(trajectory_actions_one_hot, dim=1)
            trajectory_probs = torch.cat(trajectory_probs, dim=1)
            trajectory_states = torch.cat(trajectory_states, dim=1)

            trajectory_actions = trajectory_actions.unsqueeze(2)
            trajectory_actions = trajectory_actions.permute(1, 0, 2)
            trajectory_actions_one_hot = trajectory_actions_one_hot.permute(1, 0, 2)
            trajectory_probs = trajectory_probs.permute(1, 0, 2)
            trajectory_states = trajectory_states.permute(1, 0, 2)

            selected_indices_np = np.arange(0, 2*z_seq_len*primitive_trajectory_len,
                                            2*primitive_trajectory_len)

            selected_indices = torch.from_numpy(selected_indices_np)

            trajectory_actions_selected = trajectory_actions[:, selected_indices]
            trajectory_actions_one_hot_selected = trajectory_actions_one_hot[:, selected_indices]
            trajectory_probs_selected = trajectory_probs[:, selected_indices]
            trajectory_states_selected = trajectory_states[:, selected_indices]

            if not argmax:

                trajectory_probs_selected_np = trajectory_probs_selected.clone().numpy()
                max_probs = np.max(trajectory_probs_selected_np, axis=2)
                max_probs_argsorted = max_probs.argsort(axis=1)
                arg_maxes = np.argmax(trajectory_probs_selected_np, axis=2)

                uniform_probs_minus_argmax = (1.0/(self.action_size - 1)) * \
                                              np.ones((num_trajectories, z_seq_len, self.action_size - 1))

                action_indices = multinomial_sample(uniform_probs_minus_argmax, axis=2)
                action_indices[action_indices >= arg_maxes] += 1

                trajectory_actions_selected_np = trajectory_actions_selected.clone().numpy()

                for i in range(max_probs.shape[0]):

                    low_prob = max_probs[i, max_probs_argsorted[i, low_repulsion_idx]]
                    high_prob = max_probs[i, max_probs_argsorted[i, high_repulsion_idx]]

                    for j in range(z_seq_len):

                        if (max_probs[i, j] > low_prob - 0.001) and (max_probs[i, j] < high_prob + 0.001):

                            trajectory_actions_selected_np[i, j] = action_indices[i, j]

                trajectory_actions_selected = torch.from_numpy(trajectory_actions_selected_np)
                trajectory_actions_one_hot_selected = torch.zeros(trajectory_actions_one_hot_selected.shape)
                trajectory_actions_one_hot_selected.scatter_(-1, trajectory_actions_selected, 1)

            return trajectory_actions_selected, trajectory_actions_one_hot_selected, \
                   trajectory_probs_selected, trajectory_states_selected


    def argmax_primitive_seq(self, z, initial_env_state=torch.zeros(0),
                             env=None, modify_trajectory_len=False, trajectory_len=10, reset=True):

        #z = [1, z_size]
        # output_actions = [seq_len, 1, action_size]
        # output_env_states = [seq_len, 1, env_state_size]
        # output_action_probs_seq = [seq_len, 1, action_size]

        seq_len = self.seq_len
        if modify_trajectory_len:
            seq_len = trajectory_len

        initial_env_state = initial_env_state.unsqueeze(0).unsqueeze(0)

        with torch.no_grad():

            decoder_input = initial_env_state
            next_env_state = initial_env_state

            decoder_input_with_z = torch.cat([decoder_input, z.unsqueeze(0)], -1)

            output_actions = torch.zeros(seq_len, 1, self.action_size)
            output_env_states = torch.zeros(seq_len, 1, self.env_state_size)
            output_action_probs_seq = torch.ones(seq_len, 1, self.action_size)*1E-9
            done = False

            for t in range(seq_len):

                output = self.decoder(decoder_input_with_z)

                output_logit_action = self.h1_fc_actions(output)
                output_action_probs = torch.softmax(output_logit_action, dim=2)
                output_action_probs = output_action_probs.squeeze().detach().numpy()

                arg_max = np.argmax(output_action_probs)
                argmax_action = np.zeros((1, self.action_size))
                argmax_action[0][arg_max] = 1.0

                output_actions[t] = torch.from_numpy(argmax_action)
                output_action_probs_seq[t] = torch.from_numpy(output_action_probs).unsqueeze(0)

                output_env_states[t] = next_env_state
                next_env_state, r, done, _ = env.step(arg_max)

                if done:
                    if reset:
                        next_env_state = env.reset()
                    else:
                        next_env_state = torch.from_numpy(next_env_state).float()
                        break

                next_env_state = torch.from_numpy(next_env_state).float()
                decoder_input = next_env_state.unsqueeze(0)

                decoder_input_with_z = torch.cat([decoder_input, z], -1).unsqueeze(0)

            return output_actions, output_env_states, next_env_state, \
                   output_action_probs_seq, done, t

    def argmax_trajectory_h(self, z, initial_env_state=torch.zeros(0),
                            env=None, modify_trajectory_len=False, trajectory_len=10,
                            hierarchy_level="h1", reset=False, random_traj=False, modify_top_len=False, top_len=10):

        # initial_env_state = [env_state_size]
        # z = [1, z_size]
        # output_actions = [seq_len, 1, action_size]
        # output_env_states = [seq_len, 1, env_state_size]
        # output_action_probs_seq = [seq_len, 1, action_size]

        if hierarchy_level == "h1":

            output_actions, output_env_states, next_env_state, output_action_probs_seq, \
            done, t = self.argmax_primitive_seq(z, initial_env_state=initial_env_state,
                                                env=env, modify_trajectory_len=modify_trajectory_len,
                                                trajectory_len=trajectory_len, reset=reset)

            return output_actions, output_env_states, next_env_state, output_action_probs_seq, None, done, t

        if hierarchy_level == "h2":

            if modify_top_len:
                top_seq_len = top_len
            else:
                top_seq_len = self.h2_top_seq_len

            output_actions_list = []
            output_env_states_list = []
            output_action_probs_seq_list = []
            h1_z_list = []

            next_env_state = initial_env_state
            done = False

            for i in range(top_seq_len):

                decoder_input = next_env_state.unsqueeze(0).unsqueeze(0)
                decoder_input_with_z = torch.cat([decoder_input, z.unsqueeze(0)], -1)
                output = self.decoder(decoder_input_with_z, hierarchy_level="h2")

                h1_z_raw = self.h2_fc_actions(output).squeeze(0)
                h1_z = 2.5 * torch.tanh(h1_z_raw)

                if random_traj:
                    h1_z = torch.randn_like(h1_z)

                output_actions, output_env_states, next_env_state, output_action_probs_seq, \
                _, done, t = self.argmax_trajectory_h(h1_z, next_env_state,
                                                      env=env,
                                                      modify_trajectory_len=modify_trajectory_len,
                                                      trajectory_len=trajectory_len,
                                                      hierarchy_level="h1")

                output_actions_list += [output_actions]
                output_env_states_list += [output_env_states]
                output_action_probs_seq_list += [output_action_probs_seq]
                h1_z_list += [h1_z.unsqueeze(1)]

                if done:
                    break

            last_index = self.h2_bottom_seq_len * i + t
            last_index += 1

            for j in range(i+1, top_seq_len):

                output_actions_list += [torch.zeros(self.h2_bottom_seq_len, 1, self.action_size)]
                output_env_states_list += [torch.zeros(self.h2_bottom_seq_len, 1, self.env_state_size)]
                output_action_probs_seq_list += [torch.ones(self.h2_bottom_seq_len, 1, self.action_size)*1E-9]
                h1_z_list += [torch.zeros(1, 1, self.h1_z_size)]

            output_actions_list = torch.cat(output_actions_list, dim=0)
            output_env_states_list = torch.cat(output_env_states_list, dim=0)
            output_action_probs_seq_list = torch.cat(output_action_probs_seq_list, dim=0)
            h1_z_list = torch.cat(h1_z_list, dim=0)

            return output_actions_list, output_env_states_list, next_env_state, \
                   output_action_probs_seq_list, h1_z_list, done, last_index


        if hierarchy_level == "h3":

            if modify_top_len:
                top_seq_len = top_len
            else:
                top_seq_len = self.h3_top_seq_len

            output_actions_list = []
            output_env_states_list = []
            output_action_probs_seq_list = []
            h2_z_list = []

            next_env_state = initial_env_state
            done = False

            for i in range(top_seq_len):

                decoder_input = next_env_state.unsqueeze(0).unsqueeze(0)
                decoder_input_with_z = torch.cat([decoder_input, z.unsqueeze(0)], -1)
                output = self.decoder(decoder_input_with_z, hierarchy_level="h3")

                h2_z_raw = self.h3_fc_actions(output).squeeze(0)
                h2_z = 2.5 * torch.tanh(h2_z_raw)

                if random_traj:
                    h2_z = torch.randn_like(h2_z)

                h2_z_list += [h2_z.unsqueeze(1)]

                for j in range(self.h3_bottom_seq_len):

                    decoder_input = next_env_state.unsqueeze(0).unsqueeze(0)
                    decoder_input_with_z = torch.cat([decoder_input, h2_z.unsqueeze(0)], -1)
                    output = self.decoder(decoder_input_with_z, hierarchy_level="h2")

                    h1_z_raw = self.h2_fc_actions(output).squeeze(0)
                    h1_z = 2.5 * torch.tanh(h1_z_raw)

                    output_actions, output_env_states, next_env_state, output_action_probs_seq, \
                     _, done, t = self.argmax_trajectory_h(h1_z, next_env_state,
                                                           env=env,
                                                           modify_trajectory_len=modify_trajectory_len,
                                                           trajectory_len=trajectory_len,
                                                           hierarchy_level="h1")

                    output_actions_list += [output_actions]
                    output_env_states_list += [output_env_states]
                    output_action_probs_seq_list += [output_action_probs_seq]

                    if done:
                        break

                if done:
                    break

            last_index = self.h3_bottom_seq_len*self.h2_bottom_seq_len*i + self.h2_bottom_seq_len * j + t
            last_index += 1

            for k in range(j+1, self.h3_bottom_seq_len):

                output_actions_list += [torch.zeros(self.h2_bottom_seq_len, 1, self.action_size)]
                output_env_states_list += [torch.zeros(self.h2_bottom_seq_len, 1, self.env_state_size)]
                output_action_probs_seq_list += [torch.ones(self.h2_bottom_seq_len, 1, self.action_size)*1E-9]

            for k in range(i+1, top_seq_len):

                output_actions_list += [torch.zeros(self.h3_bottom_seq_len*self.h2_bottom_seq_len, 1, self.action_size)]
                output_env_states_list += [torch.zeros(self.h3_bottom_seq_len*self.h2_bottom_seq_len, 1, self.env_state_size)]
                output_action_probs_seq_list += [torch.ones(self.h3_bottom_seq_len*self.h2_bottom_seq_len, 1, self.action_size)*1E-9]
                h2_z_list += [torch.zeros(1, 1, self.h2_z_size)]

            output_actions_list = torch.cat(output_actions_list, dim=0)
            output_env_states_list = torch.cat(output_env_states_list, dim=0)
            output_action_probs_seq_list = torch.cat(output_action_probs_seq_list, dim=0)
            h2_z_list = torch.cat(h2_z_list, dim=0)

            return output_actions_list, output_env_states_list, next_env_state, \
                   output_action_probs_seq_list, h2_z_list, done, last_index


    def repulsion_trajectory_h(self, z, initial_env_state=torch.zeros(0), env=None,
                               modify_trajectory_len=False, trajectory_len=10,
                               hierarchy_level="h2"):

        # z = [1, z_size]
        # output_actions = [seq_len, 1, action_size]
        # output_env_states = [seq_len, 1, env_state_size]
        # output_action_probs_seq = [seq_len, 1, action_size]

        with torch.no_grad():

            if hierarchy_level == "h2":

                output_actions_list = []
                output_env_states_list = []
                output_action_probs_seq_list = []
                h1_z_list = []
                repulsion_indicator_list = []

                next_env_state = initial_env_state
                done = False

                for i in range(self.h2_top_seq_len):

                    decoder_input = next_env_state.unsqueeze(0).unsqueeze(0)
                    decoder_input_with_z = torch.cat([decoder_input, z.unsqueeze(0)], -1)
                    output = self.decoder(decoder_input_with_z, hierarchy_level="h2")

                    h1_z_raw = self.h2_fc_actions(output).squeeze(0)
                    h1_z = 2.5 * torch.tanh(h1_z_raw)

                    repulsion_indicator = [0]
                    r = np.random.uniform(size=1)
                    r = r[0]
                    repulsion_prob = 0.5

                    if r < repulsion_prob:
                        repulsion_indicator[0] = 1
                        h1_z = torch.randn(1, self.h1_z_size)

                    output_actions, output_env_states, next_env_state, \
                        output_action_probs_seq, _, done, t = \
                        self.argmax_trajectory_h(h1_z, next_env_state, env=env,
                                                 modify_trajectory_len=modify_trajectory_len,
                                                 trajectory_len=trajectory_len,
                                                 hierarchy_level="h1")

                    output_actions_list += [output_actions]
                    output_env_states_list += [output_env_states]
                    output_action_probs_seq_list += [output_action_probs_seq]
                    h1_z_list += [h1_z.unsqueeze(1)]
                    repulsion_indicator_list += [repulsion_indicator]

                    if done:
                        break

                last_index = self.h2_bottom_seq_len * i + t
                last_index += 1

                for j in range(i+1, self.h2_top_seq_len):
                    output_actions_list += [torch.zeros(self.h2_bottom_seq_len, 1, self.action_size)]
                    output_env_states_list += [torch.zeros(self.h2_bottom_seq_len, 1, self.env_state_size)]
                    output_action_probs_seq_list += [torch.ones(self.h2_bottom_seq_len, 1, self.action_size) * 1E-9]
                    h1_z_list += [torch.zeros(1, 1, self.h1_z_size)]

                output_actions_list = torch.cat(output_actions_list, dim=0)
                output_env_states_list = torch.cat(output_env_states_list, dim=0)
                output_action_probs_seq_list = torch.cat(output_action_probs_seq_list, dim=0)
                h1_z_list = torch.cat(h1_z_list, dim=0)

                return output_actions_list, output_env_states_list, next_env_state, \
                       output_action_probs_seq_list, \
                       [h1_z_list, repulsion_indicator_list], done, last_index

            if hierarchy_level == "h3":

                output_actions_list = []
                output_env_states_list = []
                output_action_probs_seq_list = []
                h2_z_list = []
                repulsion_indicator_list = []

                next_env_state = initial_env_state
                done = False

                for i in range(self.h3_top_seq_len):

                    decoder_input = next_env_state.unsqueeze(0).unsqueeze(0)
                    decoder_input_with_z = torch.cat([decoder_input, z.unsqueeze(0)], -1)
                    output = self.decoder(decoder_input_with_z, hierarchy_level="h3")

                    h2_z_raw = self.h3_fc_actions(output).squeeze(0)
                    h2_z = 2.5 * torch.tanh(h2_z_raw)

                    repulsion_indicator = [0]
                    r = np.random.uniform(size=1)
                    r = r[0]
                    repulsion_prob = 0.5

                    if r < repulsion_prob:
                        repulsion_indicator[0] = 1
                        h2_z = torch.randn(1, self.h1_z_size)

                    h2_z_list += [h2_z.unsqueeze(1)]
                    repulsion_indicator_list += [repulsion_indicator]

                    for j in range(self.h3_bottom_seq_len):

                        decoder_input = next_env_state.unsqueeze(0).unsqueeze(0)
                        decoder_input_with_z = torch.cat([decoder_input, h2_z.unsqueeze(0)], -1)
                        output = self.decoder(decoder_input_with_z, hierarchy_level="h2")

                        h1_z_raw = self.h2_fc_actions(output).squeeze(0)
                        h1_z = 2.5 * torch.tanh(h1_z_raw)

                        output_actions, output_env_states, next_env_state, \
                            output_action_probs_seq, _, done, t = \
                            self.argmax_trajectory_h(h1_z, next_env_state,
                                                     env=env,
                                                     modify_trajectory_len=modify_trajectory_len,
                                                     trajectory_len=trajectory_len,
                                                     hierarchy_level="h1")

                        output_actions_list += [output_actions]
                        output_env_states_list += [output_env_states]
                        output_action_probs_seq_list += [output_action_probs_seq]

                        if done:
                            break

                    if done:
                        break

                last_index = self.h3_bottom_seq_len * self.h2_bottom_seq_len * i + self.h2_bottom_seq_len * j + t
                last_index += 1

                for k in range(j + 1, self.h3_bottom_seq_len):
                    output_actions_list += [torch.zeros(self.h2_bottom_seq_len, 1, self.action_size)]
                    output_env_states_list += [torch.zeros(self.h2_bottom_seq_len, 1, self.env_state_size)]
                    output_action_probs_seq_list += [torch.ones(self.h2_bottom_seq_len, 1, self.action_size) * 1E-9]

                for k in range(i + 1, self.h3_top_seq_len):
                    output_actions_list += [
                        torch.zeros(self.h3_bottom_seq_len * self.h2_bottom_seq_len, 1, self.action_size)]
                    output_env_states_list += [
                        torch.zeros(self.h3_bottom_seq_len * self.h2_bottom_seq_len, 1, self.env_state_size)]
                    output_action_probs_seq_list += [
                        torch.ones(self.h3_bottom_seq_len * self.h2_bottom_seq_len, 1, self.action_size) * 1E-9]
                    h2_z_list += [torch.zeros(1, 1, self.h2_z_size)]

                output_actions_list = torch.cat(output_actions_list, dim=0)
                output_env_states_list = torch.cat(output_env_states_list, dim=0)
                output_action_probs_seq_list = torch.cat(output_action_probs_seq_list, dim=0)
                h2_z_list = torch.cat(h2_z_list, dim=0)

                return output_actions_list, output_env_states_list, next_env_state, \
                       output_action_probs_seq_list, \
                       [h2_z_list, repulsion_indicator_list], done, last_index


    def argmax_or_repulsion_from_region_h(self, pos_to_sample, z_grid=None, argmax=True, env=None,
                                          hierarchy_level="h2", random_traj=False,
                                          bootstrap="no", modify_top_len=False, top_len=10, old_model=None):

        # pos_to_sample = [num_trajectories, z_size]
        # trajectory_actions = [num_trajectories, seq_len, 1]
        # trajectory_actions_one_hot = [num_trajectories, seq_len, action_size]
        # trajectory_probs = [num_trajectories, seq_len, action_size]
        # trajectory_states = [num_trajectories, seq_len, env_state_size]

        modify_trajectory_len = True
        trajectory_len = self.h2_bottom_seq_len

        with torch.no_grad():

            num_trajectories = pos_to_sample.shape[0]

            trajectory_actions = []
            trajectory_actions_one_hot = []
            trajectory_probs = []
            trajectory_states = []
            trajectory_h_zs = []
            trajectory_repulsion_indicators = []

            lengths = torch.zeros(num_trajectories)

            for i in range(num_trajectories):

                repulsion_indicators = None
                z_sample = pos_to_sample[i].unsqueeze(0)

                if bootstrap == "no":
                    initial_env_state = env.reset()
                    initial_env_state = torch.from_numpy(initial_env_state).float()

                else:

                    random_index = torch.randint(num_trajectories, (1,))[0]
                    z_sample_bootstrap = z_grid[random_index].unsqueeze(0)

                    initial_env_state = env.reset()
                    initial_env_state = torch.from_numpy(initial_env_state).float()

                    output_actions, output_env_states, next_env_state, \
                    output_probs, h_zs, done, last_index = \
                        old_model.argmax_trajectory_h(z_sample_bootstrap, initial_env_state, env=env,
                                                      hierarchy_level=bootstrap, random_traj=False,
                                                      modify_trajectory_len=modify_trajectory_len,
                                                      trajectory_len=trajectory_len,
                                                      modify_top_len=modify_top_len, top_len=top_len)

                    initial_env_state = next_env_state
                    if done:
                        initial_env_state = env.reset()
                        initial_env_state = torch.from_numpy(initial_env_state).float()

                if argmax:
                    output_actions, output_env_states, next_env_state, \
                        output_probs, h_zs, done, last_index = \
                        self.argmax_trajectory_h(z_sample, initial_env_state, env=env,
                                            hierarchy_level=hierarchy_level, random_traj=random_traj,
                                            modify_trajectory_len=modify_trajectory_len, trajectory_len=trajectory_len)
                else:
                    output_actions, output_env_states, next_env_state, \
                        output_probs, [h_zs, repulsion_indicators], done, last_index = \
                        self.repulsion_trajectory_h(z_sample, initial_env_state, env=env,
                                            hierarchy_level=hierarchy_level,
                                            modify_trajectory_len=modify_trajectory_len,
                                            trajectory_len=trajectory_len)

                classes = torch.tensor(range(self.action_size))
                outputs_class = torch.sum(classes * output_actions, dim=-1).long()

                lengths[i] = last_index

                trajectory_actions += [outputs_class]
                trajectory_actions_one_hot += [output_actions]
                trajectory_probs += [output_probs]
                trajectory_states += [output_env_states]
                trajectory_h_zs += [h_zs]
                if repulsion_indicators is not None:
                    trajectory_repulsion_indicators += [repulsion_indicators]

            trajectory_actions = torch.cat(trajectory_actions, dim=1)
            trajectory_actions_one_hot = torch.cat(trajectory_actions_one_hot, dim=1)
            trajectory_probs = torch.cat(trajectory_probs, dim=1)
            trajectory_states = torch.cat(trajectory_states, dim=1)
            trajectory_h_zs = torch.cat(trajectory_h_zs, dim=1)

            trajectory_actions = trajectory_actions.unsqueeze(2)
            trajectory_actions = trajectory_actions.permute(1, 0, 2)
            trajectory_actions_one_hot = trajectory_actions_one_hot.permute(1, 0, 2)
            trajectory_probs = trajectory_probs.permute(1, 0, 2)
            trajectory_states = trajectory_states.permute(1, 0, 2)
            trajectory_h_zs = trajectory_h_zs.permute(1, 0, 2)

            return trajectory_actions, trajectory_actions_one_hot, trajectory_probs, \
                   trajectory_h_zs, trajectory_repulsion_indicators, trajectory_states, lengths


def multinomial_sample(prob_matrix, axis=1):

    s = prob_matrix.cumsum(axis=axis)
    prob_matrix_shape = prob_matrix.shape[:-1]
    r = np.random.rand(*prob_matrix_shape, 1)
    k = (s < r).sum(axis=axis)
    return k






