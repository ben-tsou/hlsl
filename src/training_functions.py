import numpy as np
import torch
from torch._six import inf
import time

import math

from Seq2Seq_model import Seq2Seq_VAE
from lstm_models import create_mask

NLL = torch.nn.NLLLoss(reduction='sum')


def weight_tail(data, lengths, max_length, hierarch_level="h2"):

    # data = [seq_len, batch, input_size]

    assert data.shape[1] == len(lengths), "Dimension Mismatch!"
    weight = torch.ones_like(data)

    for i in range(data.shape[1]):

        length = int(lengths[i].item())

        if length < max_length:
            if hierarch_level == "h2":
                start = max(length - 7, 0)
            elif hierarch_level == "h3":
                start = max(length - 25, 0)
            weight[start:length, i, :] = -1.0

        if length <= 0:
            raise Exception("length should always be positive.  Length is: " + str(length))

    return weight


def loss_function_custom(log_probs, target_actions, mu, logvar, kl_tolerance, beta, weights=None):

    log_probs = log_probs.view(-1, log_probs.shape[-1])

    target_actions_one_hot = torch.zeros(log_probs.shape)
    target_actions_one_hot.scatter_(-1, target_actions.unsqueeze(1), 1)

    if weights is None:
        weights = torch.ones_like(target_actions_one_hot)

    probs = torch.exp(log_probs)

    epsilon = 1E-6
    modified_log_probs = torch.log(probs * weights + (1.0 - weights)/2 + epsilon)

    NLL_loss = torch.sum(target_actions_one_hot * -1.0 * modified_log_probs)

    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    KLD = torch.clamp(KLD, min=kl_tolerance)

    loss = NLL_loss + beta * KLD

    return loss, NLL_loss, KLD, beta


def loss_function_categorical(log_probs, target_actions, mu, logvar, kl_tolerance, beta):

    log_probs = log_probs.view(-1, log_probs.shape[-1])
    NLL_loss = NLL(log_probs, target_actions)

    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    KLD = torch.clamp(KLD, min=kl_tolerance)

    loss = NLL_loss + beta * KLD

    return loss, NLL_loss, KLD, beta


def kl_anneal_function(anneal_function, step, k, x0, target_beta=1.0, floor=0.5):

    if anneal_function == 'logistic':
        return float(target_beta / (1 + np.exp(-k * (step - x0))))
    elif anneal_function == 'fast_logistic':
        return float(target_beta / (1 + np.exp(-4*k * (step - x0/4))))
    elif anneal_function == 'fast_logistic2':
        return float(target_beta / (1 + np.exp(-k * (step - x0/4))))
    elif anneal_function == 'logistic_floor':
        return floor + (1-floor) * float(target_beta / (1 + np.exp(-k * (step - x0))))
    elif anneal_function == 'linear':
        return min(1, step / x0)
    else:
        return NotImplementedError


def clip_grad_norm_(parameters, max_norm, norm_type=2):
    r"""Clips gradient norm of an iterable of parameters.

    The norm is computed over all gradients together, as if they were
    concatenated into a single vector. Gradients are modified in-place.

    Arguments:
        parameters (Iterable[Tensor] or Tensor): an iterable of Tensors or a
            single Tensor that will have gradients normalized
        max_norm (float or int): max norm of the gradients
        norm_type (float or int): type of the used p-norm. Can be ``'inf'`` for
            infinity norm.

    Returns:
        Total norm of the parameters (viewed as a single vector).
    """
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    parameters = list(filter(lambda p: p.grad is not None, parameters))
    max_norm = float(max_norm)
    norm_type = float(norm_type)
    if norm_type == inf:
        total_norm = max(p.grad.data.abs().max() for p in parameters)
    else:
        total_norm = 0
        for p in parameters:
            param_norm = p.grad.data.norm(norm_type)
            total_norm += param_norm.item() ** norm_type
        total_norm = total_norm ** (1. / norm_type)
    clip_coef = max_norm / (total_norm + 1e-6)
    if clip_coef < 1:
        for p in parameters:
            p.grad.data.mul_(clip_coef)
    return total_norm

def clip_grad_value_(parameters, clip_value):
    r"""Clips gradient of an iterable of parameters at specified value.

    Gradients are modified in-place.

    Arguments:
        parameters (Iterable[Tensor] or Tensor): an iterable of Tensors or a
            single Tensor that will have gradients normalized
        clip_value (float or int): maximum allowed value of the gradients.
            The gradients are clipped in the range
            :math:`\left[\text{-clip\_value}, \text{clip\_value}\right]`
    """
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    clip_value = float(clip_value)
    for p in filter(lambda p: p.grad is not None, parameters):
        p.grad.data.clamp_(min=-clip_value, max=clip_value)


def run_VAE(model: Seq2Seq_VAE, optimizer, input_env_states, input_actions, input_actions_one_hot,
            num_epochs, clip_gn, clip_gv, batch_size,
            kl_tolerance=0, anneal='logistic', logistic_scale=10.0, target_beta=1.0, floor=0.5,
            log_interval=10, log_epoch_interval=1, f=None, writer=None,
            hierarchy_level="h1", lengths=None):

    start_time = time.time()

    num_samples = input_actions.shape[0]
    total_train_batch = math.ceil(num_samples / batch_size)

    x0 = total_train_batch * num_epochs / 2.0
    k = logistic_scale / (num_epochs * total_train_batch)

    for epoch in range(num_epochs):

        # input_actions = [batch, seq_len, 1]
        # input_actions_one_hot = [batch, seq_len, action_size]

        randomize = np.arange(num_samples)
        np.random.shuffle(randomize)
        randomize_tensor = torch.tensor(randomize)

        input_actions_shuffled = input_actions[randomize_tensor]
        input_actions_one_hot_shuffled = input_actions_one_hot[randomize_tensor]

        if hierarchy_level != "h1":
            lengths_shuffled = lengths[randomize_tensor]

        input_env_states_shuffled = input_env_states[randomize_tensor]
        input_env_states_seq = input_env_states_shuffled.permute(1, 0, 2)

        input_actions_seq = input_actions_shuffled.permute(1, 0, 2)
        input_actions_one_hot_seq = input_actions_one_hot_shuffled.permute(1, 0, 2)

        model.train()

        train_loss = 0
        train_NLLLoss = 0
        train_KLD = 0

        for i in range(total_train_batch):

            offset = (i * batch_size) % num_samples

            input_actions_one_hot_batch = input_actions_one_hot_seq[:, offset:(offset + batch_size), :]
            input_actions_batch = input_actions_seq[:, offset:(offset + batch_size), :]

            if hierarchy_level != "h1":
                lengths_batch = lengths_shuffled[offset:(offset + batch_size)]
            else:
                lengths_batch = None

            input_env_states_batch = input_env_states_seq[:, offset:(offset + batch_size), :]
            data_length = input_actions_one_hot_batch.shape[1]

            optimizer.zero_grad()

            output_log_probs, mu, logvar, batch_z = model(input_env_states_batch, input_actions_one_hot_batch,
                                                          lengths_batch, hierarchy_level=hierarchy_level)

            if hierarchy_level == "h1":

                input_actions_reshaped = input_actions_batch.reshape(-1)
                weight_reshaped = None

            else:

                mask_batch = create_mask(input_actions_batch, lengths_batch)
                mask_reshaped = mask_batch.reshape(-1)
                input_actions_reshaped = input_actions_batch.reshape(-1)
                input_actions_reshaped = input_actions_reshaped * mask_reshaped

                if hierarchy_level == "h2":
                    max_length = model.h2_bottom_seq_len * model.h2_top_seq_len
                elif hierarchy_level == "h3":
                    max_length = model.h2_bottom_seq_len * model.h3_bottom_seq_len * model.h3_top_seq_len

                weight_batch = weight_tail(input_actions_one_hot_batch, lengths_batch, max_length,
                                           hierarch_level=hierarchy_level)
                weight_reshaped = weight_batch.reshape(-1, model.action_size)

            if anneal != 'const':
                beta = kl_anneal_function(anneal, total_train_batch * epoch + i,
                                          k, x0, target_beta, floor=floor)
            else:
                beta = target_beta

            loss, NLL_loss, KLD, beta = loss_function_custom(output_log_probs, input_actions_reshaped,
                                                             mu, logvar, kl_tolerance, beta,
                                                             weights=weight_reshaped)

            if writer:
                writer.add_scalar('beta', beta, total_train_batch * epoch + i)

            loss.backward()

            filtered_parameters = list(filter(lambda p: p.grad is not None, model.parameters()))

            total_grad_norm_before_clipping = \
                torch.stack([p.grad.detach().norm(2) for p in filtered_parameters]).norm(2)

            torch.nn.utils.clip_grad_value_(model.parameters(), clip_gv)
            parameters_grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), clip_gn)

            total_grad_norm_after_clipping = \
                torch.stack([p.grad.detach().norm(2) for p in filtered_parameters]).norm(2)

            clip_coeff = total_grad_norm_after_clipping/total_grad_norm_before_clipping

            weights = []
            for param in model.parameters():
                weights.append(param.clone().detach())

            total_norm = 0
            with torch.no_grad():
                for p in model.parameters():
                    param_norm = p.detach().norm(2)
                    total_norm += param_norm.item() ** 2

            total_norm = total_norm ** (1. / 2)

            if writer:
                writer.add_scalar('parameters_norm', total_norm, total_train_batch * epoch + i)
                writer.add_scalar('parameters_grad_norm', parameters_grad_norm, total_train_batch * epoch + i)
                writer.add_scalar('parameters_grad_norm_after_clipping',
                                  total_grad_norm_after_clipping, total_train_batch * epoch + i)
                writer.add_scalar('parameters_clip_coeff', clip_coeff, total_train_batch * epoch + i)

            train_loss += loss.item()
            train_NLLLoss += NLL_loss.item()
            train_KLD += KLD.item()

            optimizer.step()

            weights_after_backprop = []
            for param in model.parameters():
                weights_after_backprop.append(param)

            total_difference_norm = 0
            total_weights_norm = 0
            total_norm_again = 0

            for j in range(len(weights)):

                difference_norm = (weights[j] - weights_after_backprop[j]).detach().norm(2)
                wab_norm = weights_after_backprop[j].detach().norm(2)
                w_norm = weights[j].detach().norm(2)

                total_difference_norm += difference_norm.item() ** 2
                total_weights_norm += wab_norm.item() ** 2
                total_norm_again += w_norm.item() ** 2

            total_difference_norm = total_difference_norm ** (1. / 2)
            total_weights_norm = total_weights_norm ** (1. / 2)
            total_norm_again = total_norm_again ** (1. / 2)

            if writer:
                writer.add_scalar('total__norm_again', total_norm_again, total_train_batch * epoch + i)
                writer.add_scalar('total_weights_norm', total_weights_norm, total_train_batch * epoch + i)
                writer.add_scalar('total_difference_norm', total_difference_norm, total_train_batch * epoch + i)
                writer.add_scalar('ratio_norm', total_difference_norm/total_weights_norm, total_train_batch * epoch + i)

            if i % log_interval == 0 and epoch % log_epoch_interval == 0:
                s = 'Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tL_likelihood: {:.6f}\tL_divergence: {:.6f}'.format(
                        epoch, i * batch_size, num_samples, 100. * i * batch_size / num_samples,
                               loss.item() / data_length, NLL_loss.item() / data_length, KLD.item() / data_length)

                if f:
                    f.write(s + '\n')

                print(s)

        if epoch % log_epoch_interval == 0:

            if f:
                f.write('====> Epoch: {} Average loss: {:.4f}'.format(epoch, train_loss / num_samples) + '\n')
            print('====> Epoch: {} Average loss: {:.4f}'.format(epoch, train_loss / num_samples))

            print("epoch time: " + str(time.time() - start_time))
            start_time = time.time()

            named_params = model.named_parameters()

            if writer:
                for name, param in named_params:
                    if "layernorm" not in name:
                        writer.add_histogram(name + '_weight_dist', param.detach(), epoch)
                        if param.grad is not None:
                            writer.add_histogram(name + '_grad_dist', param.grad.detach(), epoch)
                        writer.add_scalar(name + '_param_norm', param.detach().norm(2), epoch)
                        if param.grad is not None:
                            writer.add_scalar(name + '_grad_norm', param.grad.detach().norm(2), epoch)

        if writer:
            writer.add_scalar('training_loss', train_loss / num_samples, epoch)
            writer.add_scalar('training_NLL_loss', train_NLLLoss / num_samples, epoch)
            writer.add_scalar('training_KLD', train_KLD / num_samples, epoch)
            writer.add_scalar('learning_rate', optimizer.param_groups[0]['lr'], epoch)

    if f:
        f.write('\n\n\n\n')

    return train_loss / num_samples
