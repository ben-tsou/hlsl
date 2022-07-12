from lstm_models import *

def test_script_layerNorm():
    x = torch.tensor([[1.5, .0, .0, .0]])
    their_layerNorm_False = torch.nn.LayerNorm(4, elementwise_affine=False)
    their_layerNorm_True = torch.nn.LayerNorm(4, elementwise_affine=True)
    my_layerNorm = LayerNorm(4)
    y1_False = their_layerNorm_False(x)
    y1_True = their_layerNorm_True(x)
    y2 = my_layerNorm(x)

    print(y1_False)
    print(y1_True)
    print(y2)

    assert (y1_False - y2).abs().max() < 1e-3
    assert (y1_True - y2).abs().max() < 1e-3

    params = their_layerNorm_True.state_dict(keep_vars=True)
    named_params = their_layerNorm_True.named_parameters()

    print("Printing params: \n")

    for name, param in params.items():
        print(name)
        print(param.requires_grad)
        print(param.shape)

    print("test_script_layerNorm passed!!\n\n")


def test_script_bidir_dropout(seq_len, batch, input_size, hidden_size):

    inp = torch.randn(seq_len, batch, input_size)

    initial_state = [LSTMState(torch.randn(batch, hidden_size),
                         torch.randn(batch, hidden_size))
               for _ in range(2)]

    rnn = BidirLSTMLayer(LayerNormDropoutLSTMCell, input_size, hidden_size,
                    forget_bias=1.0, use_ff_dropout=True, ff_drop_prob=0.0, same_ff_mask=True,
                    use_recurrent_dropout=True, recurrent_drop_prob=0.5, same_recurrent_mask=True,
                    use_layer_norm=True)

    print("forget_bias: ", rnn.forward_layer.cell.forget_bias)
    print("use_ff_dropout: ", rnn.forward_layer.cell.use_ff_dropout)
    print("ff_drop_prob: ", rnn.forward_layer.cell.ff_drop_prob)
    print("same_ff_mask: ", rnn.forward_layer.cell.same_ff_mask)
    print("use_recurrent_dropout: ", rnn.forward_layer.cell.use_recurrent_dropout)
    print("recurrent_drop_prob: ", rnn.forward_layer.cell.recurrent_drop_prob)
    print("same_recurrent_mask: ", rnn.forward_layer.cell.same_recurrent_mask)
    print("use_layer_norm: ", rnn.forward_layer.cell.use_layer_norm)

    print("\nBidirLSTMLayer LayerNormLSTMCell parameters: \n")

    params = rnn.state_dict(keep_vars=True)
    named_params = rnn.named_parameters()

    print("Printing params: \n")

    for name, param in params.items():
        print(name)
        print(param.requires_grad)
        print(param.shape)

    print("\nPrinting named params: \n")

    for name, param in named_params:
        print(name)
        print(param.requires_grad)
        print(param.shape)

    print("recurrent_dropout_mask: ", rnn.forward_layer.cell.recurrent_dropout_mask)
    print("ff_dropout_mask: ", rnn.forward_layer.cell.ff_dropout_mask)

    output, out_state = rnn(inp, initial_state)

    print("recurrent_dropout_mask 2: ", rnn.forward_layer.cell.recurrent_dropout_mask)
    print("ff_dropout_mask 2: ", rnn.forward_layer.cell.ff_dropout_mask)

    output2, out_state2 = rnn(inp, initial_state, reset_dropout_masks=False)

    print("recurrent_dropout_mask 3: ", rnn.forward_layer.cell.recurrent_dropout_mask)
    print("ff_dropout_mask 3: ", rnn.forward_layer.cell.ff_dropout_mask)


    state = initial_state
    inputs = inp.unbind(0)
    outputs = []
    for i in range(len(inputs)):
        if i == 1:
            print("Inside for loop out shape: ", out.shape)
        out, state = rnn(inputs[i].unsqueeze(0), state, reset_dropout_masks=False)
        outputs += [out.squeeze()]

    print("recurrent_dropout_mask: ", rnn.forward_layer.cell.recurrent_dropout_mask)
    print("ff_dropout_mask: ", rnn.forward_layer.cell.ff_dropout_mask)

    unrolled_output = torch.stack(outputs)
    unrolled_out_state = state

    print(unrolled_output.shape)
    print(output.shape)

    print(unrolled_output[:, 1, 1])
    print(output[:, 1, 1])

    print(output[:, 1, :])
    print(output[:, 2, :])

    assert (output - output2).abs().max() < 1e-5

    assert (out_state.forward_state.cx - out_state2[0][1]).abs().max() < 1e-5
    assert (out_state.backward_state.hx - out_state2[1][0]).abs().max() < 1e-5

    print("\ntest_script_dropout passed!!\n\n")


def test_script_dropout(seq_len, batch, input_size, hidden_size):

    #input_size = hidden_size

    inp = torch.randn(seq_len, batch, input_size)

    initial_state = LSTMState(torch.randn(batch, hidden_size),
                              torch.randn(batch, hidden_size))

    rnn = LSTMLayer(LayerNormDropoutLSTMCell, input_size, hidden_size,
                    forget_bias=1.0, use_ff_dropout=True, ff_drop_prob=0.0, same_ff_mask=True,
                    use_recurrent_dropout=True, recurrent_drop_prob=0.5, same_recurrent_mask=True,
                    use_layer_norm=True)

    print("forget_bias: ", rnn.cell.forget_bias)
    print("use_ff_dropout: ", rnn.cell.use_ff_dropout)
    print("ff_drop_prob: ", rnn.cell.ff_drop_prob)
    print("same_ff_mask: ", rnn.cell.same_ff_mask)
    print("use_recurrent_dropout: ", rnn.cell.use_recurrent_dropout)
    print("recurrent_drop_prob: ", rnn.cell.recurrent_drop_prob)
    print("same_recurrent_mask: ", rnn.cell.same_recurrent_mask)
    print("use_layer_norm: ", rnn.cell.use_layer_norm)

    print("\nLSTMLayer LayerNormLSTMCell parameters: \n")

    params = rnn.state_dict(keep_vars=True)
    named_params = rnn.named_parameters()

    print("Printing params: \n")

    for name, param in params.items():
        print(name)
        print(param.requires_grad)
        print(param.shape)

    print("\nPrinting named params: \n")

    for name, param in named_params:
        print(name)
        print(param.requires_grad)
        print(param.shape)

    print("recurrent_dropout_mask: ", rnn.cell.recurrent_dropout_mask)
    print("ff_dropout_mask: ", rnn.cell.ff_dropout_mask)

    output, out_state = rnn(inp, initial_state)

    print("recurrent_dropout_mask 2: ", rnn.cell.recurrent_dropout_mask)
    print("ff_dropout_mask 2: ", rnn.cell.ff_dropout_mask)

    output2, out_state2 = rnn(inp, initial_state, reset_dropout_masks=False)

    print("recurrent_dropout_mask 3: ", rnn.cell.recurrent_dropout_mask)
    print("ff_dropout_mask 3: ", rnn.cell.ff_dropout_mask)

    print("output shape: ", output.shape)
    print("outstate length: ", len(out_state))
    print(out_state[0].shape)
    print(out_state[1].shape)

    state = initial_state
    inputs = inp.unbind(0)
    outputs = []
    for i in range(len(inputs)):
        if i == 1:
            print("Inside for loop out shape: ", out.shape)
        out, state = rnn(inputs[i].unsqueeze(0), state, reset_dropout_masks=False)
        outputs += [out.squeeze()]

    print("recurrent_dropout_mask: ", rnn.cell.recurrent_dropout_mask)
    print("ff_dropout_mask: ", rnn.cell.ff_dropout_mask)

    unrolled_output = torch.stack(outputs)
    unrolled_out_state = state

    print(unrolled_output.shape)
    print(output.shape)

    print(unrolled_output[:, 1, 1])
    print(output[:, 1, 1])

    print(output[:, 1, :])
    print(output[:, 2, :])

    #print(out[:, 0, 0])
    #print(unrolled_out[:, 0, 0])

    assert (output - unrolled_output).abs().max() < 1e-5

    assert (output - output2).abs().max() < 1e-5

    assert (out_state[0] - out_state2[0]).abs().max() < 1e-5
    assert (out_state[1] - out_state2[1]).abs().max() < 1e-5

    assert (out_state[0] - unrolled_out_state[0]).abs().max() < 1e-5
    assert (out_state[1] - unrolled_out_state[1]).abs().max() < 1e-5

    print("\ntest_script_dropout passed!!\n\n")


def test_script_residual_layer(seq_len, batch, input_size, hidden_size):

    inp = torch.randn(seq_len, batch, input_size)

    state = LSTMState(torch.randn(batch, hidden_size),
                      torch.randn(batch, hidden_size))

    rnn2 = ResidualLSTMLayer(LayerNormDropoutLSTMCell, input_size, hidden_size,
                     forget_bias=0.0, use_ff_dropout=False, use_recurrent_dropout=False, use_layer_norm=False)

    print("forget_bias: ", rnn2.cell.forget_bias)
    print("use_ff_dropout: ", rnn2.cell.use_ff_dropout)
    print("use_recurrent_dropout: ", rnn2.cell.use_recurrent_dropout)
    print("use_layer_norm: ", rnn2.cell.use_layer_norm)

    # Control: pytorch native LSTM
    lstm = nn.LSTM(input_size, hidden_size, 1)
    lstm_state = LSTMState(state.hx.unsqueeze(0), state.cx.unsqueeze(0))

    print("nn.LSTM parameters: \n")
    print("length of lstm.all_weights[0]: ", len(lstm.all_weights[0]))

    print(lstm.all_weights[0][0].shape)    #input gate
    print(lstm.all_weights[0][1].shape)    #hidden gate
    print(lstm.all_weights[0][2].shape)    #input bias
    print(lstm.all_weights[0][3].shape)    #hidden bias

    print("\n")

    print(lstm.all_weights[0][2])
    print(lstm.all_weights[0][3])

    print("\nLSTMLayer LayerNormLSTMCell parameters: \n")

    params = rnn2.state_dict()

    for name, param in params.items():
        print(name)
        print(param.shape)

    for lstm_param, custom_param2 in zip(lstm.all_weights[0],
                                        [rnn2.cell.weight_ih, rnn2.cell.weight_hh,
                                        torch.zeros(4*hidden_size), torch.zeros(4*hidden_size)] ):

        assert lstm_param.shape == custom_param2.shape

        with torch.no_grad():
            lstm_param.copy_(custom_param2)

    out2, out_state2 = rnn2(inp, state)
    lstm_out, lstm_out_state = lstm(inp, lstm_state)

    print("inp shape: ", inp.shape)
    print("out2 shape: ", out2.shape)
    print("lstm_out shape: ", lstm_out.shape)
    print("\n")

    print(inp[:, 0, 0])
    print(out2[:, 0, 0])
    print(lstm_out[:, 0, 0])

    assert (out2 - inp - lstm_out).abs().max() < 1e-5
    assert (out_state2[0] - lstm_out_state[0]).abs().max() < 1e-5
    assert (out_state2[1] - lstm_out_state[1]).abs().max() < 1e-5

    print("test_script_residual_layer passed!!\n\n")


def test_script_rnn_layer(seq_len, batch, input_size, hidden_size):

    inp = torch.randn(seq_len, batch, input_size)

    state = LSTMState(torch.randn(batch, hidden_size),
                      torch.randn(batch, hidden_size))

    rnn2 = LSTMLayer(LayerNormDropoutLSTMCell, input_size, hidden_size,
                     forget_bias=0.0, use_ff_dropout=False, use_recurrent_dropout=False, use_layer_norm=False)

    print("forget_bias: ", rnn2.cell.forget_bias)
    print("use_ff_dropout: ", rnn2.cell.use_ff_dropout)
    print("use_recurrent_dropout: ", rnn2.cell.use_recurrent_dropout)
    print("use_layer_norm: ", rnn2.cell.use_layer_norm)

    # Control: pytorch native LSTM
    lstm = nn.LSTM(input_size, hidden_size, 1)
    lstm_state = LSTMState(state.hx.unsqueeze(0), state.cx.unsqueeze(0))

    print("nn.LSTM parameters: \n")
    print("length of lstm.all_weights[0]: ", len(lstm.all_weights[0]))

    print(lstm.all_weights[0][0].shape)    #input gate
    print(lstm.all_weights[0][1].shape)    #hidden gate
    print(lstm.all_weights[0][2].shape)    #input bias
    print(lstm.all_weights[0][3].shape)    #hidden bias

    print("\n")

    print(lstm.all_weights[0][2])
    print(lstm.all_weights[0][3])

    print("\nLSTMLayer LayerNormLSTMCell parameters: \n")

    params = rnn2.state_dict()

    for name, param in params.items():
        print(name)
        print(param.shape)


    for lstm_param, custom_param2 in zip(lstm.all_weights[0],
                                        [rnn2.cell.weight_ih, rnn2.cell.weight_hh,
                                        torch.zeros(4*hidden_size), torch.zeros(4*hidden_size)] ):

        assert lstm_param.shape == custom_param2.shape

        with torch.no_grad():
            lstm_param.copy_(custom_param2)

    out2, out_state2 = rnn2(inp, state)
    lstm_out, lstm_out_state = lstm(inp, lstm_state)

    print("inp shape: ", inp.shape)
    print("state length: ", len(state))
    print("state[0] shape: ", state[0].shape)

    print("out2 shape: ", out2.shape)
    print("out_state2 length: ", len(out_state2))
    print("out_state2[0] shape: ", out_state2[0].shape)

    print("out_state2 type: ", type(out_state2))
    print(out_state2._fields)
    print(out_state2[0])
    print(out_state2.hx)
    print(out_state2.hx.shape, out_state2.cx.shape)

    assert (out2 - lstm_out).abs().max() < 1e-5
    assert (out_state2[0] - lstm_out_state[0]).abs().max() < 1e-5
    assert (out_state2[1] - lstm_out_state[1]).abs().max() < 1e-5

    print("test_script_rnn_layer passed!!\n\n")


def test_script_stacked_rnn(seq_len, batch, input_size, hidden_size,
                            num_layers):

    inp = torch.randn(seq_len, batch, input_size)

    states = [LSTMState(torch.randn(batch, hidden_size),
                        torch.randn(batch, hidden_size))
              for _ in range(num_layers)]

    forget_bias=0.0
    use_ff_dropout=False
    ff_drop_prob=0.0
    same_ff_mask=True
    use_recurrent_dropout=False
    recurrent_drop_prob=0.0
    same_recurrent_mask=True
    use_layer_norm=False

    first_layer_args = [LayerNormDropoutLSTMCell, input_size, hidden_size,
                    forget_bias, use_ff_dropout, ff_drop_prob, same_ff_mask,
                    use_recurrent_dropout, recurrent_drop_prob, same_recurrent_mask,
                    use_layer_norm]

    other_layer_args = [LayerNormDropoutLSTMCell, hidden_size, hidden_size,
                        forget_bias, use_ff_dropout, ff_drop_prob, same_ff_mask,
                        use_recurrent_dropout, recurrent_drop_prob, same_recurrent_mask,
                        use_layer_norm]


    rnn1 = StackedLSTM(num_layers, LSTMLayer, LSTMLayer,
                       first_layer_args=first_layer_args,
                       other_layer_args=other_layer_args)

    # Control: pytorch native LSTM
    lstm = nn.LSTM(input_size, hidden_size, num_layers)
    lstm_state = flatten_states(states)

    print("states transformation:")
    print(len(states))
    print(states[0][0].shape)
    print(states[0][1].shape)

    print("\nlstm_state length: ", len(lstm_state))
    print(lstm_state[0].shape)

    print("\nnn.LSTM parameters: \n")

    print(len(lstm.all_weights))
    for i in range(num_layers):
        print(lstm.all_weights[i][0].shape)   #input gate
        print(lstm.all_weights[i][1].shape)   #hidden gate
        print(lstm.all_weights[i][2].shape)   #input bias
        print(lstm.all_weights[i][3].shape)   #hidden bias

    print(lstm.all_weights[0][2])
    print(lstm.all_weights[0][3])

    print("\nLSTMLayer LayerNormLSTMCell parameters: \n")

    params = rnn1.state_dict()

    for name, param in params.items():
        print(name)
        print(param.shape)

    for layer in range(num_layers):
        custom_params1 = [rnn1.layers[layer].cell.weight_ih,
                          rnn1.layers[layer].cell.weight_hh,
                          torch.zeros(4 * hidden_size),
                          torch.zeros(4 * hidden_size)]

        for lstm_param, custom_param1 in zip(lstm.all_weights[layer], custom_params1):
            assert lstm_param.shape == custom_param1.shape

            with torch.no_grad():
                lstm_param.copy_(custom_param1)

    out1, out_state1 = rnn1(inp, states)
    custom_state1 = flatten_states(out_state1)

    print("out1 shape: ", out1.shape)
    print("out_state1 length: ", len(out_state1))
    print("out_state1[0] length: ", len(out_state1[0]))
    print("out_state1[0][0] shape: ", out_state1[0][0].shape)

    lstm_out, lstm_out_state = lstm(inp, lstm_state)

    print(len(custom_state1))
    print("custom_state1[0] shape: ", custom_state1[0].shape)

    print("num layers: ")
    print(len(custom_state1[1]))

    assert (out1 - lstm_out).abs().max() < 1e-5
    assert (custom_state1[0] - lstm_out_state[0]).abs().max() < 1e-5
    assert (custom_state1[1] - lstm_out_state[1]).abs().max() < 1e-5

    print(lstm_out.shape)
    print(lstm_out[0][0])

    print("custom_state1 shape: ")
    print(custom_state1[0].shape)

    print("lstm_out_state shape: ")
    print(lstm_out_state[0].shape)

    print("test_script_stacked_rnn passed!!\n\n")


def test_script_bidir_rnn_layer(seq_len, batch, input_size, hidden_size):

    inp = torch.randn(seq_len, batch, input_size)

    states = [LSTMState(torch.randn(batch, hidden_size),
                         torch.randn(batch, hidden_size))
               for _ in range(2)]

    rnn2 = BidirLSTMLayer(LayerNormDropoutLSTMCell, input_size, hidden_size,
                          forget_bias=0.0, use_ff_dropout=False, use_recurrent_dropout=False, use_layer_norm=False)

    print(type(rnn2.forward_layer))
    print(type(rnn2.backward_layer))

    # Control: pytorch native LSTM
    lstm = nn.LSTM(input_size, hidden_size, 1, bidirectional=True)
    lstm_state = flatten_states(states)

    print("nn.LSTM parameters: \n")
    print(len(lstm.all_weights))

    print(lstm.all_weights[0][0].shape)
    print(lstm.all_weights[0][1].shape)
    print(lstm.all_weights[0][2].shape)
    print(lstm.all_weights[0][3].shape)

    print(lstm.all_weights[0][2])
    print(lstm.all_weights[0][3])

    print("BidirLSTMLayer LayerNormDropoutLSTMCell parameters: \n")

    params = rnn2.state_dict()

    for name, param in params.items():
        print(name)
        print(param.shape)

    print(rnn2.forward_layer.cell.weight_ih)
    print(rnn2.forward_layer.cell.weight_hh)

    print(rnn2.backward_layer.cell.weight_ih)
    print(rnn2.backward_layer.cell.weight_hh)

    for lstm_param, custom_param2 in zip(lstm.all_weights[0],
                                    [rnn2.forward_layer.cell.weight_ih, rnn2.forward_layer.cell.weight_hh,
                                     torch.zeros(4*hidden_size), torch.zeros(4*hidden_size)] ):

        assert lstm_param.shape == custom_param2.shape

        with torch.no_grad():
            lstm_param.copy_(custom_param2)

    for lstm_param, custom_param2 in zip(lstm.all_weights[1],
                                    [rnn2.backward_layer.cell.weight_ih, rnn2.backward_layer.cell.weight_hh,
                                     torch.zeros(4*hidden_size), torch.zeros(4*hidden_size)] ):

        assert lstm_param.shape == custom_param2.shape

        with torch.no_grad():
            lstm_param.copy_(custom_param2)

    out2, out_state2 = rnn2(inp, states)

    print("out2 shape:")
    print(out2.shape)

    print("out_state2 shape:")
    print(out_state2._fields)
    print(out_state2.forward_state._fields)
    print(out_state2.forward_state.hx.shape)
    print(out_state2.forward_state.cx.shape)

    print(out_state2.forward_state.cx)
    print(out_state2[0][1])

    print(len(out_state2))
    print(len(out_state2[0]))
    print(out_state2[0][0].shape)

    lstm_out, lstm_out_state = lstm(inp, lstm_state)

    print(out_state2[0][0])
    print(lstm_out_state[0][0])

    print(out_state2[0][1])
    print(lstm_out_state[0][1])

    print(out_state2[1][0])
    print(lstm_out_state[1][0])

    print(out_state2[1][1])
    print(lstm_out_state[1][1])

    assert (out2 - lstm_out).abs().max() < 1e-5
    assert (out_state2[0][0] - lstm_out_state[0][0]).abs().max() < 1e-5
    assert (out_state2[0][1] - lstm_out_state[1][0]).abs().max() < 1e-5

    assert (out_state2[1][0] - lstm_out_state[0][1]).abs().max() < 1e-5
    assert (out_state2[1][1] - lstm_out_state[1][1]).abs().max() < 1e-5

    print("test_script_bidir_rnn_layer passed!!\n\n")


def test_script_bidir_residual_rnn_layer(seq_len, batch, input_size, hidden_size):

    inp = torch.randn(seq_len, batch, input_size)

    states = [LSTMState(torch.randn(batch, hidden_size),
                         torch.randn(batch, hidden_size))
               for _ in range(2)]

    rnn2 = BidirResidualLSTMLayer(LayerNormDropoutLSTMCell, input_size, hidden_size,
                          forget_bias=0.0, use_ff_dropout=False, use_recurrent_dropout=False, use_layer_norm=False)

    print(type(rnn2.forward_layer))
    print(type(rnn2.backward_layer))

    # Control: pytorch native LSTM
    lstm = nn.LSTM(input_size, hidden_size, 1, bidirectional=True)
    lstm_state = flatten_states(states)

    print("nn.LSTM parameters: \n")
    print(len(lstm.all_weights))

    print(lstm.all_weights[0][0].shape)
    print(lstm.all_weights[0][1].shape)
    print(lstm.all_weights[0][2].shape)
    print(lstm.all_weights[0][3].shape)

    print(lstm.all_weights[0][2])
    print(lstm.all_weights[0][3])

    print("BidirResidualLSTMLayer LayerNormDropoutLSTMCell parameters: \n")

    params = rnn2.state_dict()

    for name, param in params.items():
        print(name)
        print(param.shape)

    for lstm_param, custom_param2 in zip(lstm.all_weights[0],
                                    [rnn2.forward_layer.cell.weight_ih, rnn2.forward_layer.cell.weight_hh,
                                     torch.zeros(4*hidden_size), torch.zeros(4*hidden_size)] ):

        assert lstm_param.shape == custom_param2.shape

        with torch.no_grad():
            lstm_param.copy_(custom_param2)

    for lstm_param, custom_param2 in zip(lstm.all_weights[1],
                                    [rnn2.backward_layer.cell.weight_ih, rnn2.backward_layer.cell.weight_hh,
                                     torch.zeros(4*hidden_size), torch.zeros(4*hidden_size)] ):

        assert lstm_param.shape == custom_param2.shape

        with torch.no_grad():
            lstm_param.copy_(custom_param2)

    out2, out_state2 = rnn2(inp, states)

    print("out2 shape:")
    print(out2.shape)

    print("out_state2 shape:")
    print(out_state2._fields)
    print(out_state2.forward_state._fields)
    print(out_state2.forward_state.hx.shape)
    print(out_state2.forward_state.cx.shape)

    print(len(out_state2))
    print(len(out_state2[0]))
    print(out_state2[0][0].shape)

    lstm_out, lstm_out_state = lstm(inp, lstm_state)

    print(out_state2[0][0])
    print(lstm_out_state[0][0])

    print(out_state2[0][1])
    print(lstm_out_state[0][1])

    print(out_state2[1][0])
    print(lstm_out_state[1][0])

    print(out_state2[1][1])
    print(lstm_out_state[1][1])

    print("\ninp shape: ", inp.shape)
    print("out2 shape: ", out2.shape)
    print("lstm_out shape: ", lstm_out.shape)
    print("\n")

    print(inp[:, 0, 0])
    print(out2[:, 0, 0])
    print(lstm_out[:, 0, 0])

    print(inp[:, 1, 1])
    print(out2[:, 1, 1])
    print(lstm_out[:, 1, 1])

    assert (out2 - inp - lstm_out).abs().max() < 1e-5
    assert (out_state2[0][0] - lstm_out_state[0][0]).abs().max() < 1e-5
    assert (out_state2[0][1] - lstm_out_state[1][0]).abs().max() < 1e-5

    assert (out_state2[1][0] - lstm_out_state[0][1]).abs().max() < 1e-5
    assert (out_state2[1][1] - lstm_out_state[1][1]).abs().max() < 1e-5

    print("test_script_bidir_residual_rnn_layer passed!!\n\n")


def test_script_stacked_bidir_rnn(seq_len, batch, input_size, hidden_size, num_layers):

    inp = torch.randn(seq_len, batch, input_size)

    states = [[LSTMState(torch.randn(batch, hidden_size),
                         torch.randn(batch, hidden_size))
               for _ in range(2)]
              for _ in range(num_layers)]

    forget_bias=0.0
    use_ff_dropout=False
    ff_drop_prob=0.0
    same_ff_mask=True
    use_recurrent_dropout=False
    recurrent_drop_prob=0.0
    same_recurrent_mask=True
    use_layer_norm=False

    first_layer_args = [LayerNormDropoutLSTMCell, input_size, hidden_size,
                    forget_bias, use_ff_dropout, ff_drop_prob, same_ff_mask,
                    use_recurrent_dropout, recurrent_drop_prob, same_recurrent_mask,
                    use_layer_norm]

    other_layer_args = [LayerNormDropoutLSTMCell, hidden_size*2, hidden_size,
                        forget_bias, use_ff_dropout, ff_drop_prob, same_ff_mask,
                        use_recurrent_dropout, recurrent_drop_prob, same_recurrent_mask,
                        use_layer_norm]

    rnn1 = StackedLSTM(num_layers, BidirLSTMLayer, BidirLSTMLayer,
                       first_layer_args=first_layer_args,
                       other_layer_args=other_layer_args)

    print("states transformation:")
    print(len(states))
    print(states[0][0].hx.shape)
    print(states[0][0].cx.shape)

    # Control: pytorch native LSTM
    lstm = nn.LSTM(input_size, hidden_size, num_layers, bidirectional=True)
    lstm_state = double_flatten_states(states)

    print("\nlstm_state length: ", len(lstm_state))
    print(lstm_state[0].shape)

    print("nn.LSTM parameters: \n")

    print(len(lstm.all_weights))
    for i in range(2*num_layers):
        print(i)
        print(lstm.all_weights[i][0].shape)
        print(lstm.all_weights[i][1].shape)
        print(lstm.all_weights[i][2].shape)
        print(lstm.all_weights[i][3].shape)

    print(lstm.all_weights[0][2])
    print(lstm.all_weights[0][3])

    print("Stacked BidirLSTMLayer LayerNormDropoutLSTMCell parameters: \n")

    params = rnn1.state_dict()

    for name, param in params.items():
        print(name)
        print(param.shape)

    for layer in range(num_layers):

        index = 2*layer
        custom_params1 = [rnn1.layers[layer].forward_layer.cell.weight_ih,
                          rnn1.layers[layer].forward_layer.cell.weight_hh,
                          torch.zeros(4 * hidden_size),
                          torch.zeros(4 * hidden_size)]

        for lstm_param, custom_param1 in zip(lstm.all_weights[index], custom_params1):
            assert lstm_param.shape == custom_param1.shape

            with torch.no_grad():
                lstm_param.copy_(custom_param1)

        index = 2 * layer + 1
        custom_params1 = [rnn1.layers[layer].backward_layer.cell.weight_ih,
                          rnn1.layers[layer].backward_layer.cell.weight_hh,
                          torch.zeros(4 * hidden_size),
                          torch.zeros(4 * hidden_size)]

        for lstm_param, custom_param1 in zip(lstm.all_weights[index], custom_params1):
            assert lstm_param.shape == custom_param1.shape

            with torch.no_grad():
                lstm_param.copy_(custom_param1)

    out1, out_state1 = rnn1(inp, states)
    custom_state1 = double_flatten_states(out_state1)

    print("out1 shape: ", out1.shape)
    print("out_state1 length: ", len(out_state1))
    print("out_state1[0] length: ", len(out_state1[0]))
    print("out_state1[0][0] length: ", len(out_state1[0][0]))
    print("out_state1[-1][0][0] shape: ", out_state1[-1].forward_state.hx.shape)

    lstm_out, lstm_out_state = lstm(inp, lstm_state)

    print("lstm_out_state length: ", len(lstm_out_state))
    print("custom_state1 length: ", len(custom_state1))

    print("lstm_out_state[0] shape: ", lstm_out_state[0].shape)
    print("custom_state1[0] shape: ", custom_state1[0].shape)

    assert (out1 - lstm_out).abs().max() < 1e-5
    assert (custom_state1[0] - lstm_out_state[0]).abs().max() < 1e-5
    assert (custom_state1[1] - lstm_out_state[1]).abs().max() < 1e-5


def test_script_rnn_layer_MyLSTMCell(seq_len, batch, action_size, env_state_size, z_size, hidden_size):

    input_size = action_size + env_state_size + z_size

    actions = torch.randn(seq_len, batch, action_size)
    env_states = torch.randn(seq_len, batch, env_state_size)
    zs = torch.randn(seq_len, batch, z_size)

    state = LSTMState(torch.randn(batch, hidden_size),
                      torch.randn(batch, hidden_size))

    # Control: pytorch native LSTM
    lstm = nn.LSTM(input_size, hidden_size, 1)
    lstm_state = LSTMState(state.hx.unsqueeze(0), state.cx.unsqueeze(0))

    print("nn.LSTM parameters: \n")
    print("length of lstm.all_weights: ", len(lstm.all_weights))
    print("length of lstm.all_weights[0]: ", len(lstm.all_weights[0]))

    print(lstm.all_weights[0][0].shape)    #input gate
    print(lstm.all_weights[0][1].shape)    #hidden gate
    print(lstm.all_weights[0][2].shape)    #input bias
    print(lstm.all_weights[0][3].shape)    #hidden bias

    print("\n")

    print(lstm.all_weights[0][2])
    print(lstm.all_weights[0][3])


seq_len = 10
batch = 5
inp_size = 3
hid_size = 7
num_layers = 4

action_size = 1
env_state_size = 2
z_size = 3

#test_script_layerNorm()

#test_script_dropout(seq_len, batch, inp_size, hid_size)
#test_script_bidir_dropout(seq_len, batch, inp_size, hid_size)

#test_script_residual_layer(seq_len, batch, hid_size, hid_size)

#test_script_rnn_layer(seq_len, batch, inp_size, hid_size)
#test_script_stacked_rnn(seq_len, batch, inp_size, hid_size, num_layers)

test_script_bidir_rnn_layer(seq_len, batch, inp_size, hid_size)
#test_script_bidir_residual_rnn_layer(seq_len, batch, 2*hid_size, hid_size)
#test_script_stacked_bidir_rnn(seq_len, batch, inp_size, hid_size, num_layers)

#test_script_rnn_layer_MyLSTMCell(seq_len, batch, action_size, env_state_size, z_size, hid_size)