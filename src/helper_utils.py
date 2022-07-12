import torch
import numpy as np

def random_trajectories(num_trajectories, seq_len, action_size, env_state_size, env=None):

    # random_actions = [num_trajectories, seq_len, 1]
    # random_actions_one_hot = [num_trajectories, seq_len, action_size]
    # random_env_states = [num_trajectories, seq_len, env_state_size]

    random_actions = np.random.randint(action_size, size=(num_trajectories, seq_len, 1))

    random_actions = torch.from_numpy(random_actions)
    random_actions_one_hot = torch.zeros(num_trajectories, seq_len, action_size)
    random_actions_one_hot.scatter_(-1, random_actions, 1)

    random_env_states = torch.zeros(num_trajectories, seq_len, env_state_size)

    for i in range(num_trajectories):

        next_env_state = env.reset()
        done = False

        for t in range(seq_len):

            if done:
                next_env_state = env.reset()

            random_env_states[i, t] = torch.from_numpy(next_env_state)
            next_env_state, r, done, _ = env.step(random_actions[i, t, 0].item())

    return random_actions, random_actions_one_hot, random_env_states

color2num = dict(
    gray=30,
    red=31,
    green=32,
    yellow=33,
    blue=34,
    magenta=35,
    cyan=36,
    white=37,
    crimson=38
)

def colorize(string, color, bold=False, highlight=False):
    """
    Colorize a string.

    This function was originally written by John Schulman.
    """
    attr = []
    num = color2num[color]
    if highlight: num += 10
    attr.append(str(num))
    if bold: attr.append('1')
    return '\x1b[%sm%s\x1b[0m' % (';'.join(attr), string)