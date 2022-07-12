import roboschool, gym

from gym import wrappers

from Seq2Seq_model import Seq2Seq_VAE
from LSL_utils import initial_region_points

import torch
from datetime import datetime

import os
import numpy as np


def main(model_policy_file,
         main_np_random_seed=27, main_env_seed=327, main_action_seed=200, main_observation_space_seed=100):

    print("\n".join(['- ' + spec.id for spec in gym.envs.registry.all() if spec.id.startswith('Roboschool')]))

    curpath = os.path.abspath(os.curdir)
    print("Current path is: %s" % (curpath))

    env = gym.make('MyCartPoleEvaluate-v0')

    current_time = datetime.now().strftime('%b%d_%H-%M-%S')

    expt_subdir = './cartPoleVideos/' + current_time

    if not (os.path.exists(expt_subdir)):
        os.makedirs(expt_subdir)

    env = wrappers.Monitor(env, expt_subdir, video_callable=lambda episode_id: True, force=True)

    print("env: ", env)
    print("env.unwrapped: ", env.unwrapped)
    print("env.env: ", env.env)
    print(type(env))
    print(type(env.env))

    np_random_seed = main_np_random_seed
    env_seed = main_env_seed
    action_space_seed = main_action_seed
    observation_space_seed = main_observation_space_seed
    np.random.seed(np_random_seed)
    env.seed(env_seed)
    env.action_space.seed(action_space_seed)
    env.observation_space.seed(observation_space_seed)

    obs_dim = env.observation_space.shape
    act_dim = env.action_space.shape

    print("obs_dim: ", obs_dim)
    print("act_dim: ", act_dim)
    print(len(obs_dim))
    print(len(act_dim))

    print(env.observation_space)
    print(env.action_space)

    print("observation_space high: ", env.observation_space.high)
    print("observation_space low: ", env.observation_space.low)

    env_size = obs_dim[0]
    action_size = env.action_space.n

    print("env_size: ", env_size)
    print("action_size: ", action_size)

    print("loading model: " + model_policy_file)
    model: Seq2Seq_VAE = torch.load(model_policy_file)
    print(model)

    num_grid_points = 100
    z_size = 2
    device = torch.device("cpu")

    z_grid = initial_region_points(num_grid_points, z_size)
    z_grid = torch.from_numpy(z_grid).float().to(device)

    num_trajectories = z_grid.shape[0]
    print("num_trajectories: ", num_trajectories)

    modify_trajectory_len = True
    trajectory_len = model.h2_bottom_seq_len

    traj_list = list(range(num_trajectories))

    for i in traj_list:

        z_sample = z_grid[i].unsqueeze(0)

        initial_env_state = env.reset()
        initial_env_state = torch.from_numpy(initial_env_state).float()

        next_env_state = initial_env_state

        for iteration in range(5):

            output_actions, output_env_states, next_env_state, \
            _, _, done, _ = \
                model.argmax_trajectory_h(z_sample, next_env_state, env=env,
                                          hierarchy_level="h3", random_traj=False,
                                          modify_trajectory_len=modify_trajectory_len,
                                          trajectory_len=trajectory_len)

            if done:
                break

    env.close()

if __name__ == '__main__':

    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument('--model_policy_file', type=str)

    parser.add_argument('--np_random_seed', type=int, default=27)
    parser.add_argument('--env_seed', type=int, default=327)
    parser.add_argument('--action_seed', type=int, default=200)
    parser.add_argument('--observation_space_seed', type=int, default=100)

    args = parser.parse_args()

    model_policy_file = args.model_policy_file

    main_np_random_seed = args.np_random_seed
    main_env_seed = args.env_seed
    main_action_seed = args.action_seed
    main_observation_space_seed = args.observation_space_seed

    main(model_policy_file, main_np_random_seed, main_env_seed, main_action_seed, main_observation_space_seed)