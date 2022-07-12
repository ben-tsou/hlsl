from datetime import datetime
import os

import LSL_algorithm_primitive
import LSL_algorithm_h2
import LSL_algorithm_h3

def main(main_z_size=2, main_enc_hidden_size=12, main_dec_hidden_size=50, main_num_layers=1,
         main_bidir=True, main_use_layer_norm=True, main_use_ff_dropout=False, main_use_recurrent_dropout=False,
         main_ff_drop_prob=0.0, main_recurrent_drop_prob=0.0, main_dec_num_layers=3):

    num_primitive_h2_cycles = 2
    num_full_cycles = 4

    first_primitive = True
    first_h2 = True
    primitive_h2_cycles = True

    first_h3 = True
    full_cycles = True
    full_cycles_primitive = True

    if first_primitive:

        print("Running first primitive training algorithm...")
        given_time = datetime.now().strftime('%b%d_%H-%M-%S')

        main_num_compression_iterations = 7
        main_num_logistic_compression_iterations = 5

        final_model_file = LSL_algorithm_primitive.main(
             main_anneal='logistic',
             main_num_compression_iterations=main_num_compression_iterations,
             main_num_logistic_compression_iterations=main_num_logistic_compression_iterations,
             main_initial_num_epochs=3,
             main_num_const_epochs=10,
             bootstrap="no", given_time=given_time,
             main_z_size=main_z_size, main_enc_hidden_size=main_enc_hidden_size, main_dec_hidden_size=main_dec_hidden_size,
             main_num_layers=main_num_layers, main_bidir=main_bidir, main_use_layer_norm=main_use_layer_norm,
             main_use_ff_dropout=main_use_ff_dropout, main_use_recurrent_dropout=main_use_recurrent_dropout,
             main_ff_drop_prob=main_ff_drop_prob, main_recurrent_drop_prob=main_recurrent_drop_prob,
             main_dec_num_layers=main_dec_num_layers)

    if first_h2:

        print("Running first h2 training algorithm...")
        given_time = datetime.now().strftime('%b%d_%H-%M-%S')

        main_num_compression_iterations = 4
        main_num_logistic_compression_iterations = 3

        final_model_file = LSL_algorithm_h2.main(
             main_num_compression_iterations=main_num_compression_iterations,
             main_num_logistic_compression_iterations=main_num_logistic_compression_iterations,
             main_initial_num_epochs=3,
             main_num_const_epochs=10,
             bootstrap="no", given_time=given_time,
             final_model_file=final_model_file,
             old_final_model_file=None)

    print("Starting primitive h2 alternating training...")

    if primitive_h2_cycles:

        for i in range(num_primitive_h2_cycles):

            print("primitive h2 iteration: ", i)
            print("Running primitive training...")

            old_final_model_file = final_model_file

            given_time = datetime.now().strftime('%b%d_%H-%M-%S')

            main_num_compression_iterations = 3
            main_num_logistic_compression_iterations = 2

            final_model_file = LSL_algorithm_primitive.main(
                 main_num_compression_iterations=main_num_compression_iterations,
                 main_num_logistic_compression_iterations=main_num_logistic_compression_iterations,
                 main_initial_num_epochs=3,
                 main_num_const_epochs=10,
                 bootstrap="h2", given_time=given_time,
                 final_model_file=final_model_file,
                 old_final_model_file=old_final_model_file) #old_final_model_file or None

            print("primitive h2 iteration: ", i)
            print("Running h2 training...")

            given_time = datetime.now().strftime('%b%d_%H-%M-%S')

            main_num_compression_iterations = 4
            main_num_logistic_compression_iterations = 3

            final_model_file = LSL_algorithm_h2.main(
                      main_anneal='logistic',
                      main_num_compression_iterations=main_num_compression_iterations,
                      main_num_logistic_compression_iterations=main_num_logistic_compression_iterations,
                      main_initial_num_epochs=3,
                      main_num_const_epochs=10,
                      bootstrap="no", given_time=given_time,
                      final_model_file=final_model_file, old_final_model_file=None)

    if first_h3:

        print("Running first h3 training algorithm...")

        given_time = datetime.now().strftime('%b%d_%H-%M-%S')

        main_num_compression_iterations = 4
        main_num_logistic_compression_iterations = 3

        final_model_file = LSL_algorithm_h3.main(
             main_anneal='logistic',
             main_num_compression_iterations=main_num_compression_iterations,
             main_num_logistic_compression_iterations=main_num_logistic_compression_iterations,
             main_initial_num_epochs=3,
             main_num_const_epochs=10,
             bootstrap="no", given_time=given_time,
             final_model_file=final_model_file, old_final_model_file=None)

    print("Starting primitive h2 h3 alternating training...")

    for i in range(num_full_cycles):

        small_steps = False

        print("primitive h2 h3 iteration: ", i)
        print("Running primitive training...")
        num_primitive_h2_cycles2 = 1

        old_final_model_file = final_model_file

        for j in range(num_primitive_h2_cycles2):

            given_time = datetime.now().strftime('%b%d_%H-%M-%S')

            main_num_compression_iterations = 5
            main_num_logistic_compression_iterations = 2

            main_anneal = 'logistic'

            if i > 1:
                main_num_compression_iterations = 3
                main_num_logistic_compression_iterations = 0
                main_anneal = 'const'

            if small_steps:
                main_num_compression_iterations = 1
                main_num_logistic_compression_iterations = 0
                main_anneal = 'const'

            if full_cycles_primitive:
                final_model_file = LSL_algorithm_primitive.main(
                                             main_anneal=main_anneal,
                                             main_num_compression_iterations=main_num_compression_iterations,
                                             main_num_logistic_compression_iterations=main_num_logistic_compression_iterations,
                                             main_initial_num_epochs=3,
                                             main_num_const_epochs=10,
                                             bootstrap="h3", given_time=given_time,
                                             final_model_file=final_model_file,
                                             old_final_model_file=old_final_model_file, argmax_only=False,
                                             small_steps=small_steps)

            print("primitive h2 h3 iteration: ", i)
            print("Running h2 training...")

            given_time = datetime.now().strftime('%b%d_%H-%M-%S')

            main_num_compression_iterations = 5
            main_num_logistic_compression_iterations = 2

            main_anneal = 'logistic'

            if i > 1:
                main_num_compression_iterations = 3
                main_num_logistic_compression_iterations = 0
                main_anneal = 'const'

            if small_steps:
                main_num_compression_iterations = 1
                main_num_logistic_compression_iterations = 0
                main_anneal = 'const'

            final_model_file = LSL_algorithm_h2.main(
                                  main_anneal=main_anneal,
                                  main_num_compression_iterations=main_num_compression_iterations,
                                  main_num_logistic_compression_iterations=main_num_logistic_compression_iterations,
                                  main_initial_num_epochs=3,
                                  main_num_const_epochs=10,
                                  bootstrap="h3", given_time=given_time,
                                  final_model_file=final_model_file,
                                  old_final_model_file=old_final_model_file, small_steps=small_steps,
                                  consolidate=True)

        print("primitive h2 h3 iteration: ", i)
        print("Running h3 training...")

        given_time = datetime.now().strftime('%b%d_%H-%M-%S')

        main_num_compression_iterations = 7
        main_num_logistic_compression_iterations = 3

        main_anneal = 'logistic'

        if i > 1:
            main_num_compression_iterations = 5
            main_num_logistic_compression_iterations = 0
            main_anneal = 'const'

        if small_steps:
            main_num_compression_iterations = 2
            main_num_logistic_compression_iterations = 0
            main_anneal = 'const'

        final_model_file = LSL_algorithm_h3.main(
                              main_anneal=main_anneal,
                              main_num_compression_iterations=main_num_compression_iterations,
                              main_num_logistic_compression_iterations=main_num_logistic_compression_iterations,
                              main_initial_num_epochs=3,
                              main_num_const_epochs=10,
                              bootstrap="no", given_time=given_time,
                              final_model_file=final_model_file,
                              old_final_model_file=None, small_steps=small_steps, consolidate=True)


def make_directories():

    models = ["saved_models_full", "saved_models", "saved_models_h2", "saved_models_h3"]
    tensorboard_runs = ["LSL_new_runs", "LSL_new_runs_h2", "LSL_new_runs_h3"]
    VAE_stats = ["LSL_run_VAE_stats", "LSL_run_VAE_stats_h2", "LSL_run_VAE_stats_h3"]
    z_space_trajectories = ["z_space_trajectories", "z_space_trajectories_h2", "z_space_trajectories_h3"]
    z_recon_stats = ["z_recon_stats", "z_recon_stats_h2", "z_recon_stats_h3"]
    z_lengths = ["z_lengths_h2", "z_lengths_h3"]
    z_trajectory_env_states = ["z_trajectory_env_states_h2", "z_trajectory_env_states_h3"]
    z_trajectory_actions = ["z_trajectory_actions_h2", "z_trajectory_actions_h3"]

    all_dirs = models+tensorboard_runs+VAE_stats + \
               z_space_trajectories+z_recon_stats+z_lengths+z_trajectory_env_states+z_trajectory_actions

    for dir in all_dirs:

        if not (os.path.exists(dir)):
            os.makedirs(dir)


if __name__ == '__main__':

    make_directories()

    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument('--z_size', type=int, default=2)
    parser.add_argument('--enc_hidden_size', type=int, default=12)
    parser.add_argument('--dec_hidden_size', type=int, default=50)
    parser.add_argument('--num_layers', type=int, default=1)
    parser.add_argument('--bidir', default=True)
    parser.add_argument('--use_layer_norm', default=True)
    parser.add_argument('--use_ff_dropout', default=False)
    parser.add_argument('--use_recurrent_dropout', default=False)
    parser.add_argument('--ff_drop_prob', type=float, default=0.0)
    parser.add_argument('--recurrent_drop_prob', type=float, default=0.0)
    parser.add_argument('--dec_num_layers', type=int, default=3)

    args = parser.parse_args()

    main_z_size = args.z_size
    main_enc_hidden_size = args.enc_hidden_size
    main_dec_hidden_size = args.dec_hidden_size
    main_num_layers = args.num_layers
    main_bidir = args.bidir
    main_use_layer_norm = args.use_layer_norm

    main_use_ff_dropout = args.use_ff_dropout
    main_use_recurrent_dropout = args.use_recurrent_dropout
    main_ff_drop_prob = args.ff_drop_prob
    main_recurrent_drop_prob = args.recurrent_drop_prob

    main_dec_num_layers = args.dec_num_layers

    main(main_z_size=main_z_size, main_enc_hidden_size=main_enc_hidden_size, main_dec_hidden_size=main_dec_hidden_size,
         main_num_layers=main_num_layers, main_bidir=main_bidir, main_use_layer_norm=main_use_layer_norm,
         main_use_ff_dropout=main_use_ff_dropout, main_use_recurrent_dropout=main_use_recurrent_dropout,
         main_ff_drop_prob=main_ff_drop_prob, main_recurrent_drop_prob=main_recurrent_drop_prob,
         main_dec_num_layers=main_dec_num_layers)