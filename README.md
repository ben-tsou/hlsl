# Hierarchical Latent Space Learning (HLSL)

This is a pytorch implementation of [Hierarchical Latent Space Learning][paper], 
an unsupervised algorithm for 
hierarchical latent space skill discovery.  Lower levels of the hierarchy focus on learning skill primitives 
while higher levels build on these primitives to 
learn more temporally extended skills.  (See the linked paper for a discussion of the method.)

Sample [videos][videos] of the algorithm trained on the Cartpole environment are available.  

## Setup

#### 1) Setup environment

Clone the repo and create the [conda][conda] environment from the included config file:
```shell
conda env create -f env.yml
conda activate hlsl
```

#### 2) Register custom Cartpole environment

Copy the `src` files `my_cartpole_new.py` and `my_cartpole_evaluate.py` 
into the directory `gym/envs/classic_control` (next to the original `cartpole.py` file).  

Modify the file `gym/envs/classic_control/__init__.py` by adding the lines 
```
from gym.envs.classic_control.my_cartpole_new import MyCartPoleEnvNew
from gym.envs.classic_control.my_cartpole_evaluate import MyCartPoleEnvEvaluate
```

Modify the file `gym/envs/__init__.py` by adding 

```
register(
    id='MyCartPoleNew-v0',
    entry_point='gym.envs.classic_control:MyCartPoleEnvNew',
    max_episode_steps=1000,
    reward_threshold=1000.0,
)
register(
    id='MyCartPoleEvaluate-v0',
    entry_point='gym.envs.classic_control:MyCartPoleEnvEvaluate',
    max_episode_steps=2400,
    reward_threshold=1000.0,
)
```

## Usage

To train the model:  
```shell
cd HLSL
python ./src/LSL_algorithm_full.py
```
The models from the run will be saved in the directory `saved_models_full`.  To evaluate a model 
`model_policy_file.pt`, run the command  
```
python ./src/my_cartpole_policy_sampling.py --model_policy_file model_policy_file.pt
```

[paper]: https://github.com 
[videos]: https://github.com 
[conda]: https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html

