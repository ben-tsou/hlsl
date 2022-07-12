import torch
import torch.optim as optim

import numpy as np
import time

import os
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter

from Seq2Seq_model import Seq2Seq_VAE

from helper_utils import colorize
from training_functions import run_VAE

from helper_utils import random_trajectories

from LSL_utils import initial_region_points
from LSL_utils import find_good_region
from LSL_utils import set_param_grad_status

from LSL_utils import trajectory_diagnostics

import copy

import roboschool
import gym

def main(main_anneal='logistic',
         main_num_compression_iterations=30,
         main_num_logistic_compression_iterations=15,
         main_initial_num_epochs=3000,
         main_num_const_epochs=10,
         bootstrap="no", given_time="",
         final_model_file=None, old_final_model_file=None, argmax_only=False, small_steps=False,
         main_z_size=2, main_enc_hidden_size=12, main_dec_hidden_size=50, main_num_layers=1,
         main_bidir=True, main_use_layer_norm=True, main_use_ff_dropout=False, main_use_recurrent_dropout=False,
         main_ff_drop_prob=0.0, main_recurrent_drop_prob=0.0, main_dec_num_layers=3,
         main_Adam_learning_rate=5e-3, main_clip_gn=500.0, main_clip_gv=100.0, main_kl_tolerance=0.0,
         main_target_beta=1.0, main_logistic_scale=12.0, main_floor=0.5, main_num_grid_points=400,
         main_num_trajectories=240, main_batch_size=8, main_seq_len=20,
         main_env_name = "MyCartPoleNew-v0",
         main_torch_manual_seed=684, main_np_random_seed=684, main_env_seed=6328):

    torch.set_printoptions(precision=3, sci_mode=False)
    np.set_printoptions(precision=3, suppress=True)

    torch_manual_seed = main_torch_manual_seed
    np_random_seed = main_np_random_seed
    env_seed = main_env_seed

    torch.manual_seed(torch_manual_seed)
    np.random.seed(np_random_seed)

    env_name = main_env_name

    z_size = main_z_size
    enc_hidden_size = main_enc_hidden_size
    dec_hidden_size = main_dec_hidden_size
    num_layers = main_num_layers
    bidir = main_bidir
    use_layer_norm = main_use_layer_norm

    use_ff_dropout = main_use_ff_dropout
    use_recurrent_dropout = main_use_recurrent_dropout
    ff_drop_prob = main_ff_drop_prob
    recurrent_drop_prob = main_recurrent_drop_prob

    dec_num_layers = main_dec_num_layers

    seq_len = main_seq_len

    env = gym.make(env_name)
    env.seed(env_seed)
    obs_dim = env.observation_space.shape

    env_state_size = obs_dim[0]
    action_size = env.action_space.n

    Adam_learning_rate = main_Adam_learning_rate
    clip_gn = main_clip_gn
    clip_gv = main_clip_gv
    kl_tolerance = main_kl_tolerance
    target_beta = main_target_beta
    logistic_scale = main_logistic_scale

    anneal = main_anneal
    floor = main_floor

    num_grid_points = main_num_grid_points
    num_trajectories = main_num_trajectories
    batch_size = main_batch_size

    current_time = datetime.now().strftime('%b%d_%H-%M-%S')

    model_file = ('saved_models/model' + anneal + str(torch_manual_seed) + current_time + '.pt')

    num_compression_iterations = main_num_compression_iterations

    trajectory_actions, trajectory_actions_one_hot, trajectory_env_states = \
        random_trajectories(num_trajectories, seq_len, action_size, env_state_size, env=env)

    primitive_old_model_file = old_final_model_file
    primitive_model_file = final_model_file

    if bootstrap != "no":
        load_model = True
    else:
        load_model = False

    if load_model:

        print("loading model: " + primitive_model_file)
        model: Seq2Seq_VAE = torch.load(primitive_model_file)
        print(model)

        new_model = copy.deepcopy(model)

        if primitive_old_model_file is not None:
            print("loading model: " + primitive_old_model_file)
            model: Seq2Seq_VAE = torch.load(primitive_old_model_file)
            print(model)

        new_model.h2_bottom_seq_len = 7
        new_model.h2_top_seq_len = 10

        new_model.h3_bottom_seq_len = 7
        new_model.h3_top_seq_len = 10

    else:

        print("creating new model")

        num_hierarchies = 3

        model = Seq2Seq_VAE(action_size, env_state_size, z_size, enc_hidden_size, dec_hidden_size,
                            num_layers, dec_num_layers=dec_num_layers, seq_len=seq_len,
                            bidirectional_encoder=bidir,
                            forget_bias=1.0,
                            use_ff_dropout=use_ff_dropout, ff_drop_prob=ff_drop_prob, same_ff_mask=True,
                            use_recurrent_dropout=use_recurrent_dropout, recurrent_drop_prob=recurrent_drop_prob,
                            same_recurrent_mask=True,
                            use_layer_norm=use_layer_norm, num_hierarchies=num_hierarchies)

        new_model = copy.deepcopy(model)

    device = "cpu"

    h1_encoder_params = []
    h1_decoder_params = []

    named_params = list(new_model.named_parameters())

    index = 0

    for name, param in named_params:

        print("i: " + str(index) + " ", name, param.shape)
        index += 1

        if "h1" in name:
            if "encode" in name:
                h1_encoder_params += [param]
            elif "decode" in name:
                h1_decoder_params += [param]
            else:
                h1_decoder_params += [param]

    h_param_groups = [{'params': h1_encoder_params, 'g_name': 'h1_encoder_params'},
                      {'params': h1_decoder_params, 'g_name': 'h1_decoder_params'},
                      ]

    new_optimizer = optim.AdamW(h_param_groups, lr=Adam_learning_rate, weight_decay=0.0)

    set_param_grad_status(new_optimizer, ['h1_encoder_params', 'h1_decoder_params'], [])

    if bootstrap == "no":
        num_epochs = main_initial_num_epochs
    else:
        num_epochs = anneal_logistic_epochs(0, bootstrap=bootstrap)

    current_time = datetime.now().strftime('%b%d_%H-%M-%S')

    z_recon_stats_file = ('z_recon_stats/' + 'full_' + current_time + '.txt')

    if bootstrap != "no":

        z_seq_len = seq_len
        z_grid = initial_region_points(num_grid_points, z_size)
        z_grid = torch.from_numpy(z_grid).float().to(device)

        if small_steps:
            pos_to_sample = torch.cat([z_grid]*4)
        else:
            pos_to_sample = z_grid

        trajectory_actions, trajectory_actions_one_hot, _, trajectory_env_states = \
            new_model.argmax_or_repulsion_primitive_from_region_bootstrap(pos_to_sample, z_grid, z_seq_len,
                                                            argmax=True, env=env,
                                                            bootstrap_level=bootstrap, old_model=model)

    for compress_iteration in range(num_compression_iterations):

        print(colorize("compress_iteration: ", 'red'), compress_iteration)
        print("num_epochs: ", num_epochs)

        log_dir = os.path.join('LSL_new_runs', current_time + '_' +
                               anneal + '_num_epochs_' + str(num_epochs) +
                               "comp_iter_" + str(compress_iteration) + '_' + str(torch_manual_seed))

        print(log_dir)

        run_VAE_statsfile = ('LSL_run_VAE_stats/' + 'full_' +
                             anneal + '_num_epochs_' + str(num_epochs) +
                             str(torch_manual_seed) + "_comp_iter_" + str(compress_iteration) + current_time + '.txt')

        print(colorize("Writing stats to: " + run_VAE_statsfile, 'green'))
        f = open(run_VAE_statsfile, 'a')

        writer = SummaryWriter(log_dir)

        if small_steps:
            num_epochs = 3
            anneal = 'const'

        run_VAE(new_model, new_optimizer, trajectory_env_states, trajectory_actions, trajectory_actions_one_hot,
                num_epochs=num_epochs, clip_gn=clip_gn, clip_gv=clip_gv, batch_size=batch_size,
                kl_tolerance=kl_tolerance, anneal=anneal, logistic_scale=logistic_scale, target_beta=target_beta,
                floor=floor, log_interval=1, log_epoch_interval=1, f=f, writer=writer)

        iteration_model_file = ('saved_models_full/model_h1'
                                + anneal + 'comp_iter' + str(compress_iteration) +
                                str(torch_manual_seed) + current_time + '.pt')

        torch.save(new_model, iteration_model_file)

        if compress_iteration == num_compression_iterations - 1:

            final_model_file = ('saved_models_full/model_h1_final' +
                                anneal + 'comp_iter' + str(compress_iteration) +
                                str(torch_manual_seed) + given_time + '.pt')

            torch.save(new_model, final_model_file)

        if compress_iteration == 0 or compress_iteration == num_compression_iterations - 1:
            f.close()
            writer.close()

        print(colorize("-----Finished VAE iteration-----\n", 'red'))

        z_grid = initial_region_points(num_grid_points, z_size)
        z_grid = torch.from_numpy(z_grid).float().to(device)

        z_seq_len = seq_len
        primitive_trajectory_len = 5

        num_samples = z_grid.shape[0]
        new_model.eval()

        pos_to_sample = find_good_region(new_model, z_grid, z_seq_len, primitive_trajectory_len,
                                         num_samples=num_samples, env=env,
                                         bootstrap=bootstrap, old_model=model)

        pos_to_sample = pos_to_sample[:num_samples]

        if small_steps:
            pos_to_sample = torch.cat([pos_to_sample]*4)

        print(colorize("-----Finished finding sample region-----\n", 'red'))
        new_model.eval()

        if bootstrap == "no":

            trajectory_actions_argmax, trajectory_actions_one_hot_argmax, \
                trajectory_probs_argmax, trajectory_states_argmax = \
                new_model.argmax_or_repulsion_primitive_from_region(pos_to_sample, z_seq_len, primitive_trajectory_len,
                                                                    argmax=True, env=env)
        else:
            trajectory_actions_argmax, trajectory_actions_one_hot_argmax, \
                trajectory_probs_argmax, trajectory_states_argmax = \
                new_model.argmax_or_repulsion_primitive_from_region_bootstrap(pos_to_sample, z_grid, z_seq_len,
                                                                              argmax=True, env=env,
                                                                              bootstrap_level=bootstrap,
                                                                              old_model=model)

        print(colorize("-----Finished argmax sampling-----\n", 'red'))

        start_time = time.time()

        if bootstrap == "no" and not small_steps:
            high_trajectory_actions_repulsion, high_trajectory_actions_one_hot_repulsion, \
            high_trajectory_probs_repulsion, high_trajectory_states_repulsion = \
                new_model.argmax_or_repulsion_primitive_from_region(pos_to_sample, z_seq_len, primitive_trajectory_len,
                                                                    low_repulsion_idx=7, high_repulsion_idx=9,
                                                                    argmax=False, env=env)

            mid_trajectory_actions_repulsion, mid_trajectory_actions_one_hot_repulsion, \
            mid_trajectory_probs_repulsion, mid_trajectory_states_repulsion = \
                new_model.argmax_or_repulsion_primitive_from_region(pos_to_sample, z_seq_len, primitive_trajectory_len,
                                                                    low_repulsion_idx=3, high_repulsion_idx=6,
                                                                    argmax=False, env=env)

            low_trajectory_actions_repulsion, low_trajectory_actions_one_hot_repulsion, \
            low_trajectory_probs_repulsion, low_trajectory_states_repulsion = \
                new_model.argmax_or_repulsion_primitive_from_region(pos_to_sample, z_seq_len, primitive_trajectory_len,
                                                                    low_repulsion_idx=0, high_repulsion_idx=2,
                                                                    argmax=False, env=env)
        elif not small_steps:
            high_trajectory_actions_repulsion, high_trajectory_actions_one_hot_repulsion, \
            high_trajectory_probs_repulsion, high_trajectory_states_repulsion = \
                new_model.argmax_or_repulsion_primitive_from_region_bootstrap(
                                                            pos_to_sample, z_grid, z_seq_len,
                                                            low_repulsion_idx=7, high_repulsion_idx=9,
                                                            argmax=False, env=env,
                                                            bootstrap_level=bootstrap, old_model=model)

            mid_trajectory_actions_repulsion, mid_trajectory_actions_one_hot_repulsion, \
            mid_trajectory_probs_repulsion, mid_trajectory_states_repulsion = \
                new_model.argmax_or_repulsion_primitive_from_region_bootstrap(pos_to_sample, z_grid, z_seq_len,
                                                                    low_repulsion_idx=3, high_repulsion_idx=6,
                                                                    argmax=False, env=env,
                                                                    bootstrap_level=bootstrap, old_model=model)

            low_trajectory_actions_repulsion, low_trajectory_actions_one_hot_repulsion, \
            low_trajectory_probs_repulsion, low_trajectory_states_repulsion = \
                new_model.argmax_or_repulsion_primitive_from_region_bootstrap(pos_to_sample, z_grid, z_seq_len,
                                                                    low_repulsion_idx=0, high_repulsion_idx=2,
                                                                    argmax=False, env=env,
                                                                    bootstrap_level=bootstrap, old_model=model)

        print("repulsion time: " + str(time.time() - start_time))
        print(colorize("-----Finished repulsion sampling-----\n", 'red'))

        num_trajectories = trajectory_actions_argmax.shape[0]

        trajectory_diagnostics(model=new_model, z_space_folder='z_space_trajectories', env=env,
                               epoch=compress_iteration,
                               z_grid=z_grid[:num_trajectories],
                               z_seq_len=z_seq_len, primitive_trajectory_len=5,
                               plot=True, show_plot=False,
                               z_recon_stats_file=z_recon_stats_file)

        if compress_iteration < main_num_logistic_compression_iterations and (argmax_only is False):

            num_epochs = anneal_logistic_epochs(compress_iteration, bootstrap="no")

            trajectory_actions = torch.cat([trajectory_actions_argmax,
                                            high_trajectory_actions_repulsion,
                                            mid_trajectory_actions_repulsion,
                                            low_trajectory_actions_repulsion])

            trajectory_actions_one_hot = torch.cat([trajectory_actions_one_hot_argmax,
                                                    high_trajectory_actions_one_hot_repulsion,
                                                    mid_trajectory_actions_one_hot_repulsion,
                                                    low_trajectory_actions_one_hot_repulsion])

            trajectory_env_states = torch.cat([trajectory_states_argmax,
                                               high_trajectory_states_repulsion,
                                               mid_trajectory_states_repulsion,
                                               low_trajectory_states_repulsion])
        else:

            num_epochs = main_num_const_epochs
            anneal = 'const'

            trajectory_actions = trajectory_actions_argmax
            trajectory_actions_one_hot = trajectory_actions_one_hot_argmax
            trajectory_env_states = trajectory_states_argmax

        if not small_steps:

            random_indices = np.arange(num_trajectories)
            np.random.shuffle(random_indices)

    print(colorize("-----Finished final VAE iteration-----\n", 'red'))
    new_model.eval()

    torch.save(new_model, model_file)
    new_model = torch.load(model_file)
    new_model.eval()

    return final_model_file


def anneal_logistic_epochs(compress_iteration, bootstrap="no"):

    if bootstrap == "no":
        if compress_iteration < 5:
            return 5
        if compress_iteration < 10:
            return 8
        if compress_iteration < 15:
            return 8

        return 8
    else:
        return 5


