import torch

import math
import scipy
import numpy as np
import scipy.stats
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

from Seq2Seq_model import Seq2Seq_VAE

from datetime import datetime


def trajectory_diagnostics(model: Seq2Seq_VAE = None, z_space_folder='z_space_trajectories', env=None, epoch=0,
                           z_grid=None, z_seq_len=10, primitive_trajectory_len=5,
                           plot=False, show_plot=False, z_recon_stats_file=None,
                           hierarchy_level="h1", z_lengths_file=None):

    current_rng_state_np = np.random.get_state()
    current_rng_state_torch = torch.get_rng_state()

    with torch.no_grad():

        if hierarchy_level == "h1":

            trajectory_actions_argmax, trajectory_actions_one_hot_argmax, \
            trajectory_probs_argmax, trajectory_states_argmax = \
                model.argmax_or_repulsion_primitive_from_region(z_grid, z_seq_len, primitive_trajectory_len,
                                                                argmax=True, env=env)

            argmax_lengths = None

        else:

            trajectory_actions_argmax, trajectory_actions_one_hot_argmax, \
            trajectory_probs_argmax, trajectory_h_zs_argmax, _, trajectory_states_argmax, argmax_lengths = \
                model.argmax_or_repulsion_from_region_h(z_grid, argmax=True, env=env,
                                                        hierarchy_level=hierarchy_level)

            f = open(z_lengths_file, 'a')
            mean_length = torch.mean(argmax_lengths)
            f.write("sort_trajectories_by_count argmax_lengths shape: " + str(argmax_lengths.shape) + "\n")
            f.write("sort_trajectories_by_count average length: " + str(mean_length) + "\n")
            f.write(str(argmax_lengths) + "\n")
            f.close()

        trajectory_actions_argmax = trajectory_actions_argmax.permute(1, 0, 2)
        trajectory_actions_one_hot_argmax = trajectory_actions_one_hot_argmax.permute(1, 0, 2)
        trajectory_states_argmax = trajectory_states_argmax.permute(1, 0, 2)

        mus, logvars, batch_z = model.seq_to_z(trajectory_states_argmax, trajectory_actions_one_hot_argmax,
                                               lengths=argmax_lengths, hierarchy_level=hierarchy_level)

        output = model.z_and_input_to_seq(trajectory_states_argmax, mus, hierarchy_level=hierarchy_level)
        #output = [seq_len, batch_size, hidden_size]

        seq_len = output.shape[0]
        output = output.permute(1, 0, 2)
        output = torch.reshape(output, [-1, model.h1_dec_hidden_size])

        output_probs = torch.softmax(model.h1_fc_actions(output), dim=-1)
        output_probs = torch.reshape(output_probs, [-1, seq_len, model.action_size])

        trajectory_actions_argmax = trajectory_actions_argmax.permute(1, 0, 2)
        trajectory_states_argmax = trajectory_states_argmax.permute(1, 0, 2)

        f = open(z_recon_stats_file, 'a')
        f.write("\n\ncompression_iteration: " + str(epoch) + '\n\n')
        num_trajectories = trajectory_actions_argmax.shape[0]
        for i in range(num_trajectories):
            print("i: " + str(i) + "    " +
                  str(z_grid[i]) + "   " + str(mus[i]))
            print(str(output_probs[i]))
            print(str(trajectory_actions_argmax[i]))
            print(str(trajectory_states_argmax[i]))
            f.write("i: " + str(i) + "    " +
                    str(z_grid[i]) + "   " + str(mus[i]) + '\n')
            f.write(str(output_probs[i]) + '\n')
            f.write(str(trajectory_actions_argmax[i]) + '\n')
            f.write(str(trajectory_states_argmax[i]) + '\n')

        f.write('\n\n')
        f.close()

        with torch.no_grad():
            sigmas = torch.exp(logvars / 2.0).numpy()

    if plot:

        fig = plt.figure(0, figsize=(5, 5))
        ax = fig.add_subplot(111, aspect='equal')
        my_colors = ['red', 'black', 'green', 'blue', 'yellow', 'cyan', 'orange', 'violet', 'turquoise', 'gold',
                     'lightsalmon', 'brown', 'gray', 'olive', 'lime', 'orangered', 'plum', 'fuchsia', 'hotpink',
                     'deeppink', 'crimson', 'darkmagenta', 'powderblue', 'aqua', 'dodgerblue', 'beige', 'lawngreen']
        ells = []

        for i in range(mus.shape[0]):
            ells += [Ellipse(xy=mus[i], width=sigmas[i, 0] * 2, height=sigmas[i, 1] * 2,
                             edgecolor=my_colors[1], fill=False)]

        for e in [*ells]:
            ax.add_artist(e)
            e.set_clip_box(ax.bbox)

        ax.set_xlim(-4, 4)
        ax.set_ylim(-4, 4)

        if show_plot:
            plt.show()

        current_time = datetime.now().strftime('%b%d_%H-%M-%S')
        fig.savefig(z_space_folder + '/' + '_' +
                    current_time + 'compiter_' + str(epoch) + ".png")

        plt.close(fig)

    np.random.set_state(current_rng_state_np)
    torch.set_rng_state(current_rng_state_torch)



def initial_region_points(num_grid_points, z_size):

    # z_grid = [num_grid_points, z_size]

    num = int(math.pow(num_grid_points, 1.0/z_size))

    percentiles = np.linspace(0.01, 0.99, num)

    z_coord = scipy.stats.norm.ppf(percentiles, loc=0, scale=1)
    z_coord_array = [z_coord]*z_size

    grid = np.meshgrid(*z_coord_array, sparse=False, indexing='ij')
    a = np.stack(grid, axis=-1)
    z_grid = a.reshape([-1, z_size])

    return z_grid


def set_param_grad_status(optimizer, on_param_group_names, off_param_group_names):

    for param_group in [group for group in optimizer.param_groups if group['g_name'] in off_param_group_names]:
        for p in param_group['params']:
            p.requires_grad = False
            p.grad = None

    for param_group in [group for group in optimizer.param_groups if group['g_name'] in on_param_group_names]:
        for p in param_group['params']:
            p.requires_grad = True


def find_good_region(model: Seq2Seq_VAE, z_grid, z_seq_len, primitive_trajectory_len,
                     num_samples=100, env=None,
                     hierarchy_level="h1", lengths_metric=False, bootstrap="no", old_model=None):

    KLD, lengths = diversity_metric(model, z_grid, z_seq_len, primitive_trajectory_len, env=env,
                                    hierarchy_level=hierarchy_level, bootstrap=bootstrap, old_model=old_model)

    KLD = KLD.numpy()
    region_pos = z_grid.cpu()

    if hierarchy_level == "h1" or lengths_metric is False:
        sample_weight_np = np.exp(KLD)
    else:
        if hierarchy_level == "h2":
            lower_const = 50
        elif hierarchy_level == "h3":
            lower_const = 350

        lengths = lengths.numpy()
        lower_bound = min(np.percentile(lengths, 50), lower_const)
        lengths_truncated = np.maximum(lengths - lower_bound, 0.0)
        sample_weight_np = np.exp(KLD) * lengths_truncated * lengths_truncated

    sample_weight_total = np.sum(sample_weight_np)

    # generate new samples
    sample_weight_np = sample_weight_np * num_samples / sample_weight_total
    floor_weight_np = np.floor(sample_weight_np)
    remainder_weight_np = sample_weight_np - floor_weight_np
    random_numbers = np.random.uniform(0, 1, len(sample_weight_np))

    pos_to_sample = []

    for j in range(len(sample_weight_np)):
        if int(floor_weight_np[j]) > 0:
            multiple_pos = [region_pos[j]] * int(floor_weight_np[j])
            pos_to_sample += multiple_pos
        if random_numbers[j] < remainder_weight_np[j]:
            pos_to_sample.append(region_pos[j])

    pos_to_sample = torch.stack(pos_to_sample)

    return pos_to_sample


def diversity_metric(model: Seq2Seq_VAE, z_grid, z_seq_len, primitive_trajectory_len, env=None,
                     hierarchy_level="h1", bootstrap="no", old_model=None):

    with torch.no_grad():

        if hierarchy_level == "h1":
            lengths = None

            if bootstrap == "no":
                _, trajectory_actions_one_hot_argmax, \
                _, trajectory_states_argmax = \
                    model.argmax_or_repulsion_primitive_from_region(z_grid, z_seq_len, primitive_trajectory_len,
                                                                        argmax=True, env=env)
            else:
                _, trajectory_actions_one_hot_argmax, \
                _, trajectory_states_argmax = \
                    model.argmax_or_repulsion_primitive_from_region_bootstrap(z_grid, z_grid,
                                                                    z_seq_len, primitive_trajectory_len,
                                                                    argmax=True, env=env,
                                                                    bootstrap_level=bootstrap, old_model=old_model)

        else:
            modify_top_len = True
            top_len = 3
            _, trajectory_actions_one_hot_argmax, \
            _, _, _, trajectory_states_argmax, lengths = \
                model.argmax_or_repulsion_from_region_h(z_grid, z_grid,
                                                        argmax=True, env=env,
                                                        hierarchy_level=hierarchy_level, bootstrap=bootstrap,
                                                        modify_top_len=modify_top_len, top_len=top_len,
                                                        old_model=old_model)

        trajectory_actions_one_hot_argmax = trajectory_actions_one_hot_argmax.permute(1, 0, 2)
        trajectory_states_argmax = trajectory_states_argmax.permute(1, 0, 2)

        mus, logvars, batch_z = model.seq_to_z(trajectory_states_argmax, trajectory_actions_one_hot_argmax, lengths,
                                               hierarchy_level=hierarchy_level)

        KLD = -0.5 * torch.sum(1 + logvars - mus.pow(2) - logvars.exp(), dim=1)

        return KLD, lengths


def length_filtering_logistic(lengths, argmax_lengths, trajectory_actions, trajectory_actions_one_hot,
                              trajectory_env_states, max_length, length_percentile_cutoff, global_cutoff_length=None,
                              ratios=None):

    lengths_shape = lengths.shape[0]
    lengths_np = lengths.clone().numpy()

    argmax_lengths_shape = argmax_lengths.shape[0]
    argmax_lengths_np = argmax_lengths.clone().numpy()

    cutoff_length = np.percentile(argmax_lengths_np, length_percentile_cutoff)

    if global_cutoff_length is not None:
        cutoff_length = min(cutoff_length, global_cutoff_length)

    cutoff_ratio = cutoff_length / max_length

    argmax_bool_indices = np.full(lengths_shape, False, dtype=bool)
    argmax_bool_indices[:argmax_lengths_shape] = True
    repulsion_bool_indices = np.full(lengths_shape, False, dtype=bool)
    repulsion_bool_indices[argmax_lengths_shape:] = True

    if cutoff_ratio < 0.5:
        argmax_ratio_max = 0.25
    elif cutoff_ratio < 0.75:
        argmax_ratio_max = 0.5
    elif cutoff_ratio < 0.9:
        argmax_ratio_max = 0.75
    else:
        argmax_ratio_max = 0.9

    if ratios == "h3_2":

        if cutoff_ratio < 0.5:
            argmax_ratio_max = 1.0 / 8
        elif cutoff_ratio < 0.75:
            argmax_ratio_max = 1.0 / 4
        elif cutoff_ratio < 0.9:
            argmax_ratio_max = 1.0 / 3
        else:
            argmax_ratio_max = 0.5

    lengths_indices_argmax_bool_np = (lengths_np >= cutoff_length) & argmax_bool_indices
    argmax_cutoff_indices = lengths_indices_argmax_bool_np.nonzero()[0]

    cutoff_lengths = lengths_np[argmax_cutoff_indices]
    sorted_cutoff_lengths_indices = np.argsort(cutoff_lengths)
    sorted_argmax_cutoff_indices = argmax_cutoff_indices[sorted_cutoff_lengths_indices]

    lengths_indices_repulsion_bool_np = (lengths_np >= cutoff_length) & repulsion_bool_indices
    repulsion_cutoff_indices = lengths_indices_repulsion_bool_np.nonzero()[0]

    num_argmax_cutoff_indices = argmax_cutoff_indices.shape[0]
    num_repulsion_cutoff_indices = repulsion_cutoff_indices.shape[0]
    argmax_ratio = num_argmax_cutoff_indices / (num_argmax_cutoff_indices + num_repulsion_cutoff_indices)

    num_argmax_keep = int(min(argmax_ratio_max / argmax_ratio, 1.0) * num_argmax_cutoff_indices)

    sorted_argmax_cutoff_indices_truncated = sorted_argmax_cutoff_indices[::-1][:num_argmax_keep]

    lengths_indices_bool_np = (lengths_np >= cutoff_length)
    lengths_indices_bool_np[:argmax_lengths_shape] = False

    lengths_indices_bool_np[sorted_argmax_cutoff_indices_truncated] = True
    lengths_indices_bool = torch.from_numpy(lengths_indices_bool_np)

    trajectory_actions = trajectory_actions[lengths_indices_bool]
    trajectory_actions_one_hot = trajectory_actions_one_hot[lengths_indices_bool]
    trajectory_env_states = trajectory_env_states[lengths_indices_bool]
    lengths = lengths[lengths_indices_bool]

    return lengths, trajectory_actions, trajectory_actions_one_hot, trajectory_env_states, cutoff_length


def length_filtering_const(argmax_lengths, trajectory_actions_argmax, trajectory_actions_one_hot_argmax,
                           trajectory_states_argmax, length_percentile_cutoff, global_cutoff_length=None):

    argmax_lengths_np = argmax_lengths.clone().numpy()
    cutoff_length = np.percentile(argmax_lengths_np, length_percentile_cutoff)

    if global_cutoff_length is not None:
        cutoff_length = min(cutoff_length, global_cutoff_length)

    lengths_indices_argmax_bool_np = (argmax_lengths_np >= cutoff_length)
    argmax_lengths_indices_bool = torch.from_numpy(lengths_indices_argmax_bool_np)

    trajectory_actions_argmax = trajectory_actions_argmax[argmax_lengths_indices_bool]
    trajectory_actions_one_hot_argmax = trajectory_actions_one_hot_argmax[argmax_lengths_indices_bool]

    trajectory_states_argmax = trajectory_states_argmax[argmax_lengths_indices_bool]
    argmax_lengths = argmax_lengths[argmax_lengths_indices_bool]

    return argmax_lengths, trajectory_actions_argmax, trajectory_actions_one_hot_argmax, trajectory_states_argmax, \
           cutoff_length


def length_filtering_h2_3(lengths, argmax_lengths2, trajectory_actions, trajectory_actions_one_hot,
                          trajectory_env_states, max_length, length_percentile_cutoff):

    lengths_shape = lengths.shape[0]
    lengths_np = lengths.clone().numpy()

    argmax_lengths2_shape = argmax_lengths2.shape[0]
    argmax_lengths2_np = argmax_lengths2.clone().numpy()

    cutoff_length = np.percentile(argmax_lengths2_np, length_percentile_cutoff)
    cutoff_ratio = cutoff_length / max_length

    argmax_bool_indices2 = np.full(lengths_shape, False, dtype=bool)
    argmax_bool_indices2[:argmax_lengths2_shape] = True
    repulsion_bool_indices2 = np.full(lengths_shape, False, dtype=bool)
    repulsion_bool_indices2[argmax_lengths2_shape:] = True

    if cutoff_ratio < 0.5:
        argmax_ratio_max = 1.0 / 8
    elif cutoff_ratio < 0.75:
        argmax_ratio_max = 1.0 / 4
    elif cutoff_ratio < 0.9:
        argmax_ratio_max = 1.0 / 3
    else:
        argmax_ratio_max = 0.5

    lengths_indices_argmax_bool2_np = (lengths_np >= cutoff_length) & argmax_bool_indices2
    argmax_cutoff_indices2 = lengths_indices_argmax_bool2_np.nonzero()[0]
    lengths_indices_repulsion_bool2_np = (lengths_np >= cutoff_length) & repulsion_bool_indices2
    repulsion_cutoff_indices2 = lengths_indices_repulsion_bool2_np.nonzero()[0]

    num_argmax_cutoff_indices2 = argmax_cutoff_indices2.shape[0]
    num_repulsion_cutoff_indices2 = repulsion_cutoff_indices2.shape[0]
    argmax_ratio = num_argmax_cutoff_indices2 / (num_argmax_cutoff_indices2 + num_repulsion_cutoff_indices2)

    num_argmax_keep = int(min(argmax_ratio_max / argmax_ratio, 1.0) * num_argmax_cutoff_indices2)

    random_indices = np.random.choice(num_argmax_cutoff_indices2, size=num_argmax_keep, replace=False)

    argmax_cutoff_indices_thinned2 = argmax_cutoff_indices2[random_indices]

    lengths_indices_bool_np = (lengths_np >= cutoff_length)
    lengths_indices_bool_np[:argmax_lengths2_shape] = False
    lengths_indices_bool_np[argmax_cutoff_indices_thinned2] = True

    lengths_indices_bool = torch.from_numpy(lengths_indices_bool_np)

    trajectory_actions = trajectory_actions[lengths_indices_bool]
    trajectory_actions_one_hot = trajectory_actions_one_hot[lengths_indices_bool]

    trajectory_env_states = trajectory_env_states[lengths_indices_bool]
    lengths = lengths[lengths_indices_bool]

    return lengths, trajectory_actions, trajectory_actions_one_hot, trajectory_env_states