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

from LSL_utils import initial_region_points
from LSL_utils import find_good_region

from LSL_utils import trajectory_diagnostics

from LSL_utils import length_filtering_logistic
from LSL_utils import length_filtering_const
from LSL_utils import set_param_grad_status

import copy

import roboschool
import gym

def main(main_anneal='logistic',
         main_num_compression_iterations=30,
         main_num_logistic_compression_iterations=15,
         main_initial_num_epochs=3000,
         main_num_const_epochs=10,
         bootstrap="no", given_time="",
         final_model_file=None, old_final_model_file=None,
         small_steps=False, consolidate=False,
         main_Adam_learning_rate=5e-3, main_clip_gn=500.0, main_clip_gv=100.0, main_kl_tolerance=0.0,
         main_target_beta=1.0, main_logistic_scale=12.0, main_floor=0.5, main_num_grid_points=400,
         main_batch_size=8, main_seq_len=20,
         main_env_name="MyCartPoleNew-v0",
         main_torch_manual_seed=616, main_np_random_seed=616, main_env_seed=6331):

    torch.set_printoptions(precision=3, sci_mode=False)
    np.set_printoptions(precision=3, suppress=True)

    torch.set_num_threads(1)
    print("num_threads: ", torch.get_num_threads())

    torch_manual_seed = main_torch_manual_seed
    np_random_seed = main_np_random_seed
    env_seed = main_env_seed

    torch.manual_seed(torch_manual_seed)
    np.random.seed(np_random_seed)

    env_name = main_env_name
    seq_len = main_seq_len

    env = gym.make(env_name)
    env.seed(env_seed)

    Adam_learning_rate = main_Adam_learning_rate
    clip_gn = main_clip_gn
    clip_gv = main_clip_gv
    kl_tolerance = main_kl_tolerance
    target_beta = main_target_beta
    logistic_scale = main_logistic_scale
    anneal = main_anneal
    floor = main_floor

    num_grid_points = main_num_grid_points
    batch_size = main_batch_size

    model_file = ('saved_models_h2/model' +
                  anneal + str(torch_manual_seed) + '.pt')

    num_compression_iterations = main_num_compression_iterations

    current_time = datetime.now().strftime('%b%d_%H-%M-%S')

    primitive_old_model_file = old_final_model_file
    primitive_model_file = final_model_file

    z_recon_stats_file = ('z_recon_stats_h2/' + 'full_' + current_time + '.txt')
    z_lengths_file = ('z_lengths_h2/' + 'full_' + current_time + '.txt')

    z_trajectory_actions_file = ('z_trajectory_actions_h2/' + 'full_' + current_time + '.txt')
    z_trajectory_env_states_file = ('z_trajectory_env_states_h2/' + 'full_' + current_time + '.txt')

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

    device = "cpu"

    h1_encoder_params = []
    h1_decoder_params = []

    h2_encoder_params = []
    h2_decoder_params = []

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
        elif "h2" in name:
            if "encode" in name:
                h2_encoder_params += [param]
            elif "decode" in name:
                h2_decoder_params += [param]
            else:
                h2_decoder_params += [param]

    h_param_groups = [{'params': h1_encoder_params, 'g_name': 'h1_encoder_params'},
                      {'params': h1_decoder_params, 'g_name': 'h1_decoder_params'},
                      {'params': h2_encoder_params, 'g_name': 'h2_encoder_params'},
                      {'params': h2_decoder_params, 'g_name': 'h2_decoder_params'},
                      ]

    new_optimizer = optim.AdamW(h_param_groups, lr=Adam_learning_rate, weight_decay=0.0)

    set_param_grad_status(new_optimizer, ['h2_encoder_params', 'h2_decoder_params'],
                                         ['h1_encoder_params', 'h1_decoder_params'])

    num_epochs = main_initial_num_epochs

    current_time = datetime.now().strftime('%b%d_%H-%M-%S')

    z_size = new_model.h1_z_size
    z_grid = initial_region_points(num_grid_points, z_size)
    z_grid = torch.from_numpy(z_grid).float().to(device)

    for compress_iteration in range(num_compression_iterations):

        print(colorize("compress_iteration: ", 'red'), compress_iteration)
        print("num_epochs: ", num_epochs)

        log_dir = os.path.join('LSL_new_runs_h2', current_time + '_' + 'n' +
                               anneal + '_num_epochs_' + str(num_epochs) +
                               "comp_iter_" + str(compress_iteration) + '_' + str(torch_manual_seed))

        print(log_dir)

        run_VAE_statsfile = ('LSL_run_VAE_stats_h2/' + 'full_' +
                             anneal + '_num_epochs_' + str(num_epochs) +
                             str(torch_manual_seed) +
                             "_comp_iter_" + str(compress_iteration) + current_time + '.txt')

        print(colorize("Writing stats to: " + run_VAE_statsfile, 'green'))

        z_seq_len = seq_len
        primitive_trajectory_len = 5
        num_samples = z_grid.shape[0]

        if compress_iteration == 0 and bootstrap == "no":
            pos_to_sample = z_grid

        else:
            if compress_iteration < 15:
                lengths_metric = True
            else:
                lengths_metric = False

            new_model.eval()
            pos_to_sample = find_good_region(new_model, z_grid, z_seq_len, primitive_trajectory_len,
                                             num_samples=num_samples, env=env,
                                             hierarchy_level="h2", lengths_metric=lengths_metric,
                                             bootstrap=bootstrap, old_model=model)

            pos_to_sample = pos_to_sample[:num_samples]

        if small_steps:
            pos_to_sample = torch.cat([pos_to_sample] * 4)

        print(colorize("-----Finished finding sample region-----\n", 'red'))
        new_model.eval()

        modify_top_len = True
        top_len_h2 = 10

        if bootstrap == "h3":
            top_len_h3_first = 4
            top_len_h3_second = 7

        if compress_iteration == 0 and bootstrap == "no":

            trajectory_actions_argmax, trajectory_actions_one_hot_argmax, \
            _, _, _, trajectory_states_argmax, argmax_lengths = \
                new_model.argmax_or_repulsion_from_region_h(pos_to_sample, argmax=True, env=env,
                                                            hierarchy_level="h2", random_traj=True)
        else:

            trajectory_actions_argmax, trajectory_actions_one_hot_argmax, \
            _, _, _, trajectory_states_argmax, argmax_lengths = \
                new_model.argmax_or_repulsion_from_region_h(pos_to_sample, argmax=True, env=env,
                                                            hierarchy_level="h2",
                                                            bootstrap="no",
                                                            modify_top_len=False, top_len=0)

            if bootstrap != "no":

                trajectory_actions_argmax2, trajectory_actions_one_hot_argmax2, \
                _, _, _, trajectory_states_argmax2, argmax_lengths2 = \
                    new_model.argmax_or_repulsion_from_region_h(pos_to_sample, z_grid, argmax=True, env=env,
                                                                hierarchy_level="h2", bootstrap="h2",
                                                                modify_top_len=modify_top_len, top_len=top_len_h2,
                                                                old_model=model)

                if bootstrap == "h3":

                    trajectory_actions_argmax3, trajectory_actions_one_hot_argmax3, \
                    _, _, _, trajectory_states_argmax3, argmax_lengths3 = \
                        new_model.argmax_or_repulsion_from_region_h(pos_to_sample, z_grid, argmax=True, env=env,
                                                                    hierarchy_level="h2", bootstrap=bootstrap,
                                                                    modify_top_len=modify_top_len,
                                                                    top_len=top_len_h3_first,
                                                                    old_model=model)

                    trajectory_actions_argmax4, trajectory_actions_one_hot_argmax4, \
                    _, _, _, trajectory_states_argmax4, argmax_lengths4 = \
                        new_model.argmax_or_repulsion_from_region_h(pos_to_sample, z_grid, argmax=True, env=env,
                                                                    hierarchy_level="h2", bootstrap=bootstrap,
                                                                    modify_top_len=modify_top_len,
                                                                    top_len=top_len_h3_second,
                                                                    old_model=model)

        print(colorize("-----Finished argmax sampling-----\n", 'red'))

        start_time = time.time()

        if compress_iteration > 0:

            first_trajectory_actions_repulsion, first_trajectory_actions_one_hot_repulsion, \
            _, _, _, first_trajectory_states_repulsion, first_repulsion_lengths = \
                new_model.argmax_or_repulsion_from_region_h(pos_to_sample, z_grid,
                                                            argmax=False, env=env,
                                                            hierarchy_level="h2")

            second_trajectory_actions_repulsion, second_trajectory_actions_one_hot_repulsion, \
            _, _, _, second_trajectory_states_repulsion, second_repulsion_lengths = \
                new_model.argmax_or_repulsion_from_region_h(pos_to_sample, z_grid,
                                                            argmax=False, env=env,
                                                            hierarchy_level="h2")

            third_trajectory_actions_repulsion, third_trajectory_actions_one_hot_repulsion, \
            _, _, _, third_trajectory_states_repulsion, third_repulsion_lengths = \
                new_model.argmax_or_repulsion_from_region_h(pos_to_sample, z_grid,
                                                            argmax=False, env=env,
                                                            hierarchy_level="h2")

            if bootstrap != "no":

                if bootstrap == "h2" or bootstrap == "h3":

                    first_trajectory_actions_repulsion2, first_trajectory_actions_one_hot_repulsion2, \
                        _, _, _, first_trajectory_states_repulsion2, first_repulsion_lengths2 = \
                        new_model.argmax_or_repulsion_from_region_h(pos_to_sample, z_grid,
                                                                    argmax=False, env=env,
                                                                    hierarchy_level="h2", bootstrap="h2",
                                                                    modify_top_len=modify_top_len, top_len=top_len_h2,
                                                                    old_model=model)

                    second_trajectory_actions_repulsion2, second_trajectory_actions_one_hot_repulsion2, \
                        _, _, _, second_trajectory_states_repulsion2, second_repulsion_lengths2 = \
                        new_model.argmax_or_repulsion_from_region_h(pos_to_sample, z_grid,
                                                                    argmax=False, env=env,
                                                                    hierarchy_level="h2", bootstrap="h2",
                                                                    modify_top_len=modify_top_len, top_len=top_len_h2,
                                                                    old_model=model)

                    third_trajectory_actions_repulsion2, third_trajectory_actions_one_hot_repulsion2, \
                        _, _, _, third_trajectory_states_repulsion2, third_repulsion_lengths2 = \
                        new_model.argmax_or_repulsion_from_region_h(pos_to_sample, z_grid,
                                                                    argmax=False, env=env,
                                                                    hierarchy_level="h2", bootstrap="h2",
                                                                    modify_top_len=modify_top_len, top_len=top_len_h2,
                                                                    old_model=model)

                if bootstrap == "h3":

                    first_trajectory_actions_repulsion3, first_trajectory_actions_one_hot_repulsion3, \
                    _, _, _, first_trajectory_states_repulsion3, first_repulsion_lengths3 = \
                        new_model.argmax_or_repulsion_from_region_h(pos_to_sample, z_grid,
                                                                    argmax=False, env=env,
                                                                    hierarchy_level="h2", bootstrap=bootstrap,
                                                                    modify_top_len=modify_top_len,
                                                                    top_len=top_len_h3_first,
                                                                    old_model=model)

                    second_trajectory_actions_repulsion3, second_trajectory_actions_one_hot_repulsion3, \
                    _, _, _, second_trajectory_states_repulsion3, second_repulsion_lengths3 = \
                        new_model.argmax_or_repulsion_from_region_h(pos_to_sample, z_grid,
                                                                    argmax=False, env=env,
                                                                    hierarchy_level="h2", bootstrap=bootstrap,
                                                                    modify_top_len=modify_top_len,
                                                                    top_len=top_len_h3_first,
                                                                    old_model=model)

                    third_trajectory_actions_repulsion3, third_trajectory_actions_one_hot_repulsion3, \
                    _, _, _, third_trajectory_states_repulsion3, third_repulsion_lengths3 = \
                        new_model.argmax_or_repulsion_from_region_h(pos_to_sample, z_grid,
                                                                    argmax=False, env=env,
                                                                    hierarchy_level="h2", bootstrap=bootstrap,
                                                                    modify_top_len=modify_top_len,
                                                                    top_len=top_len_h3_first,
                                                                    old_model=model)


                    first_trajectory_actions_repulsion4, first_trajectory_actions_one_hot_repulsion4, \
                    _, _, _, first_trajectory_states_repulsion4, first_repulsion_lengths4 = \
                        new_model.argmax_or_repulsion_from_region_h(pos_to_sample, z_grid,
                                                                    argmax=False, env=env,
                                                                    hierarchy_level="h2", bootstrap=bootstrap,
                                                                    modify_top_len=modify_top_len,
                                                                    top_len=top_len_h3_second,
                                                                    old_model=model)

                    second_trajectory_actions_repulsion4, second_trajectory_actions_one_hot_repulsion4, \
                    _, _, _, second_trajectory_states_repulsion4, second_repulsion_lengths4 = \
                        new_model.argmax_or_repulsion_from_region_h(pos_to_sample, z_grid,
                                                                    argmax=False, env=env,
                                                                    hierarchy_level="h2", bootstrap=bootstrap,
                                                                    modify_top_len=modify_top_len,
                                                                    top_len=top_len_h3_second,
                                                                    old_model=model)

                    third_trajectory_actions_repulsion4, third_trajectory_actions_one_hot_repulsion4, \
                    _, _, _, third_trajectory_states_repulsion4, third_repulsion_lengths4 = \
                        new_model.argmax_or_repulsion_from_region_h(pos_to_sample, z_grid,
                                                                    argmax=False, env=env,
                                                                    hierarchy_level="h2", bootstrap=bootstrap,
                                                                    modify_top_len=modify_top_len,
                                                                    top_len=top_len_h3_second,
                                                                    old_model=model)

        print("repulsion time: " + str(time.time() - start_time))

        print(colorize("-----Finished repulsion sampling-----\n", 'red'))

        num_trajectories = trajectory_actions_argmax.shape[0]

        if compress_iteration < main_num_logistic_compression_iterations and \
                (compress_iteration > 0):

            num_epochs = 8
            anneal = 'logistic'

            trajectory_actions = torch.cat([trajectory_actions_argmax,
                                            first_trajectory_actions_repulsion,
                                            second_trajectory_actions_repulsion,
                                            third_trajectory_actions_repulsion])

            trajectory_actions_one_hot = torch.cat([trajectory_actions_one_hot_argmax,
                                                    first_trajectory_actions_one_hot_repulsion,
                                                    second_trajectory_actions_one_hot_repulsion,
                                                    third_trajectory_actions_one_hot_repulsion])

            trajectory_env_states = torch.cat([trajectory_states_argmax,
                                               first_trajectory_states_repulsion,
                                               second_trajectory_states_repulsion,
                                               third_trajectory_states_repulsion])

            lengths = torch.cat(
                [argmax_lengths, first_repulsion_lengths, second_repulsion_lengths, third_repulsion_lengths])

            if bootstrap != "no":

                trajectory_actions2 = torch.cat([trajectory_actions_argmax2,
                                                 first_trajectory_actions_repulsion2,
                                                 second_trajectory_actions_repulsion2,
                                                 third_trajectory_actions_repulsion2])

                trajectory_actions_one_hot2 = torch.cat([trajectory_actions_one_hot_argmax2,
                                                         first_trajectory_actions_one_hot_repulsion2,
                                                         second_trajectory_actions_one_hot_repulsion2,
                                                         third_trajectory_actions_one_hot_repulsion2])

                trajectory_env_states2 = torch.cat([trajectory_states_argmax2,
                                                    first_trajectory_states_repulsion2,
                                                    second_trajectory_states_repulsion2,
                                                    third_trajectory_states_repulsion2])

                lengths2 = torch.cat(
                    [argmax_lengths2, first_repulsion_lengths2, second_repulsion_lengths2, third_repulsion_lengths2])

                if bootstrap == "h3":

                    trajectory_actions3 = torch.cat([trajectory_actions_argmax3,
                                                     first_trajectory_actions_repulsion3,
                                                     second_trajectory_actions_repulsion3,
                                                     third_trajectory_actions_repulsion3])

                    trajectory_actions_one_hot3 = torch.cat([trajectory_actions_one_hot_argmax3,
                                                             first_trajectory_actions_one_hot_repulsion3,
                                                             second_trajectory_actions_one_hot_repulsion3,
                                                             third_trajectory_actions_one_hot_repulsion3])

                    trajectory_env_states3 = torch.cat([trajectory_states_argmax3,
                                                        first_trajectory_states_repulsion3,
                                                        second_trajectory_states_repulsion3,
                                                        third_trajectory_states_repulsion3])

                    lengths3 = torch.cat(
                        [argmax_lengths3, first_repulsion_lengths3, second_repulsion_lengths3, third_repulsion_lengths3])


                    trajectory_actions4 = torch.cat([trajectory_actions_argmax4,
                                                     first_trajectory_actions_repulsion4,
                                                     second_trajectory_actions_repulsion4,
                                                     third_trajectory_actions_repulsion4])

                    trajectory_actions_one_hot4 = torch.cat([trajectory_actions_one_hot_argmax4,
                                                             first_trajectory_actions_one_hot_repulsion4,
                                                             second_trajectory_actions_one_hot_repulsion4,
                                                             third_trajectory_actions_one_hot_repulsion4])


                    trajectory_env_states4 = torch.cat([trajectory_states_argmax4,
                                                        first_trajectory_states_repulsion4,
                                                        second_trajectory_states_repulsion4,
                                                        third_trajectory_states_repulsion4])

                    lengths4 = torch.cat(
                        [argmax_lengths4, first_repulsion_lengths4, second_repulsion_lengths4, third_repulsion_lengths4])

            max_length = new_model.h2_top_seq_len * new_model.h2_bottom_seq_len

            #starting the filtering
            lengths, trajectory_actions, trajectory_actions_one_hot, trajectory_env_states, global_cutoff_length = \
                length_filtering_logistic(lengths, argmax_lengths, trajectory_actions, trajectory_actions_one_hot,
                                          trajectory_env_states, max_length, 50, global_cutoff_length=None)

            if bootstrap != "no":

                lengths2, trajectory_actions2, trajectory_actions_one_hot2, trajectory_env_states2, _ = \
                    length_filtering_logistic(lengths2, argmax_lengths2, trajectory_actions2, trajectory_actions_one_hot2,
                                              trajectory_env_states2, max_length, 80, global_cutoff_length)


                if bootstrap == "h3":

                    lengths3, trajectory_actions3, trajectory_actions_one_hot3, trajectory_env_states3, _ = \
                        length_filtering_logistic(lengths3, argmax_lengths3, trajectory_actions3,
                                                  trajectory_actions_one_hot3,
                                                  trajectory_env_states3, max_length, 80, global_cutoff_length)

                    lengths4, trajectory_actions4, trajectory_actions_one_hot4, trajectory_env_states4, _ = \
                        length_filtering_logistic(lengths4, argmax_lengths4, trajectory_actions4,
                                                  trajectory_actions_one_hot4,
                                                  trajectory_env_states4, max_length, 80, global_cutoff_length)

            if bootstrap != "no":

                trajectory_actions = torch.cat([trajectory_actions, trajectory_actions2])
                trajectory_actions_one_hot = torch.cat([trajectory_actions_one_hot, trajectory_actions_one_hot2])

                trajectory_env_states = torch.cat([trajectory_env_states, trajectory_env_states2])
                lengths = torch.cat([lengths, lengths2])

                if bootstrap == "h3":

                    trajectory_actions = torch.cat([trajectory_actions, trajectory_actions3, trajectory_actions4])
                    trajectory_actions_one_hot = torch.cat(
                       [trajectory_actions_one_hot, trajectory_actions_one_hot3, trajectory_actions_one_hot4])

                    trajectory_env_states = torch.cat([trajectory_env_states,
                                                       trajectory_env_states3, trajectory_env_states4])

                    lengths = torch.cat([lengths, lengths3, lengths4])

        else:

            num_epochs = main_num_const_epochs
            anneal = 'const'

            argmax_lengths, trajectory_actions_argmax, trajectory_actions_one_hot_argmax, trajectory_states_argmax, \
                     global_cutoff_length = length_filtering_const(argmax_lengths, trajectory_actions_argmax,
                                            trajectory_actions_one_hot_argmax, trajectory_states_argmax, 50)

            if bootstrap != "no":

                argmax_lengths2, trajectory_actions_argmax2, trajectory_actions_one_hot_argmax2, trajectory_states_argmax2, \
                _ = length_filtering_const(argmax_lengths2, trajectory_actions_argmax2,
                                                              trajectory_actions_one_hot_argmax2,
                                                              trajectory_states_argmax2, 80, global_cutoff_length)

                if bootstrap == "h3":

                    argmax_lengths3, trajectory_actions_argmax3, trajectory_actions_one_hot_argmax3, trajectory_states_argmax3, \
                    _ = length_filtering_const(argmax_lengths3, trajectory_actions_argmax3,
                                                                  trajectory_actions_one_hot_argmax3,
                                                                  trajectory_states_argmax3, 80, global_cutoff_length)

                    argmax_lengths4, trajectory_actions_argmax4, trajectory_actions_one_hot_argmax4, trajectory_states_argmax4, \
                    _ = length_filtering_const(argmax_lengths4, trajectory_actions_argmax4,
                                                                  trajectory_actions_one_hot_argmax4,
                                                                  trajectory_states_argmax4, 80, global_cutoff_length)

            trajectory_actions = trajectory_actions_argmax
            trajectory_actions_one_hot = trajectory_actions_one_hot_argmax
            trajectory_env_states = trajectory_states_argmax
            lengths = argmax_lengths

            if bootstrap != "no":

                trajectory_actions = torch.cat([trajectory_actions, trajectory_actions_argmax2])
                trajectory_actions_one_hot = torch.cat([trajectory_actions_one_hot,
                                                        trajectory_actions_one_hot_argmax2])

                trajectory_env_states = torch.cat([trajectory_env_states, trajectory_states_argmax2])
                lengths = torch.cat([lengths, argmax_lengths2])

                if bootstrap == "h3":

                    trajectory_actions = torch.cat([trajectory_actions,
                                                    trajectory_actions_argmax3, trajectory_actions_argmax4])

                    trajectory_actions_one_hot = torch.cat([trajectory_actions_one_hot,
                                                            trajectory_actions_one_hot_argmax3,
                                                            trajectory_actions_one_hot_argmax4])

                    trajectory_env_states = torch.cat([trajectory_env_states,
                                                       trajectory_states_argmax3, trajectory_states_argmax4])

                    lengths = torch.cat([lengths, argmax_lengths3, argmax_lengths4])

        f = open(run_VAE_statsfile, 'a')
        writer = SummaryWriter(log_dir)

        if compress_iteration == 0:
            num_epochs = 3
            anneal = 'logistic'

        if consolidate:
            anneal = 'const'

        if small_steps:
            num_epochs = 10
            anneal = 'const'

        print("num_epochs: ", num_epochs)

        run_VAE(new_model, new_optimizer, trajectory_env_states, trajectory_actions, trajectory_actions_one_hot,
                num_epochs=num_epochs,
                clip_gn=clip_gn, clip_gv=clip_gv, batch_size=batch_size,
                kl_tolerance=kl_tolerance, anneal=anneal, logistic_scale=logistic_scale, target_beta=target_beta,
                floor=floor,
                log_interval=1, log_epoch_interval=1,
                f=f, writer=writer,
                hierarchy_level="h2", lengths=lengths)

        iteration_model_file = ('saved_models_full/model_h2' +
                                 anneal +
                                'comp_iter' + str(compress_iteration) + '_' +
                                str(torch_manual_seed) + current_time + '.pt')

        torch.save(new_model, iteration_model_file)

        if compress_iteration == num_compression_iterations - 1:

            final_model_file = ('saved_models_full/model_h2_final'
                                 + anneal + 'comp_iter' + str(compress_iteration) + '_' +
                                 str(torch_manual_seed) + given_time + '.pt')

            torch.save(new_model, final_model_file)

        if compress_iteration == 0 or compress_iteration == num_compression_iterations - 1:
            f.close()
            writer.close()

        print(colorize("-----Finished VAE iteration-----\n", 'red'))

        f = open(z_trajectory_actions_file, 'a')
        f.write("\n\niteration: " + str(compress_iteration) + "\n")
        f.write("trajectory_actions_one_hot shape: " + str(trajectory_actions_one_hot.shape) + "\n")
        for i in range(trajectory_actions_one_hot.shape[0]):
            f.write("i: " + str(i) + " ")
            for j in range(trajectory_actions_one_hot.shape[1]):
                if int(trajectory_actions_one_hot[i, j, 0].item()) + \
                        int(trajectory_actions_one_hot[i, j, 1].item()) == 0:
                    f.write("z")
                else:
                    f.write(str(int(trajectory_actions_one_hot[i, j, 1].item())))
            f.write("\n\n")
        f.close()

        f = open(z_trajectory_env_states_file, 'a')
        f.write("\n\niteration: " + str(compress_iteration) + "\n")
        f.write("trajectory_env_states shape: " + str(trajectory_env_states.shape) + "\n")
        for i in range(trajectory_env_states.shape[0]):
            f.write("i: " + str(i) + "\n")
            for j in range(trajectory_env_states.shape[1]):
                if j == 0 or not torch.equal(trajectory_env_states[i, j], trajectory_env_states[i, j-1]):
                    f.write(str(trajectory_env_states[i, j]) + "\n")
            f.write("\n\n")
        f.close()

        f = open(z_lengths_file, 'a')
        f.write("\n\niteration: " + str(compress_iteration) + "\n")
        mean_length = torch.mean(lengths)
        f.write("lengths shape: " + str(lengths.shape) + "\n")
        f.write("average length: " + str(mean_length) + "\n")
        f.write(str(lengths) + "\n")
        f.close()

        trajectory_diagnostics(model=new_model, z_space_folder='z_space_trajectories_h2', env=env,
                               epoch=compress_iteration,
                               z_grid=z_grid[:num_trajectories],
                               z_seq_len=z_seq_len, primitive_trajectory_len=5,
                               plot=True, show_plot=False,
                               z_recon_stats_file=z_recon_stats_file, hierarchy_level="h2",
                               z_lengths_file=z_lengths_file)

    print(colorize("-----Finished final VAE iteration-----\n", 'red'))
    new_model.eval()

    torch.save(new_model, model_file)
    new_model = torch.load(model_file)
    new_model.eval()

    return final_model_file