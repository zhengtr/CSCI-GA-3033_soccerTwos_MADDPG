# Cooperative Multi-Agents with Reinforcement Learning 
## CSCI-GA 3033-090 Final Project 
(Deep Reinforcement Learning, Fall 2021, NYU Courant)

This is the repository of CSCI-GA 3033-090 final project. 

Created by `Tanran Zheng` (tz408@nyu.edu), `Xinru Xu` (xx2085@nyu.edu).

# Setup
You can install or conda enviroment using the conda_env.yaml file

# Implementation
- Independent Learners with DQN 
- Parameter Sharing with PPO 
- Improved version of Multi-Agents DDPG (MADDPG by Lowe et al. 2017).

All models are implemented using Pytorch.

## To run
To train PPO and DQN model:
```
python dqn_ppo/train.py
```
To train MADDPG model:
```
python maddpg/train.py
```

## To check the result
1. Logger is activated and will store the training/evaluation log in `./exp_local/`;
2. Package `wandb` is included in the code. The training/evaluation curve can be shown on `wandb`;


## Detailed report
See `Final Report - Cooperative Multi-Agents with Reinforcement Learning.pdf`, which is written in blog style as recommended. 

Video demonstration included.


## Reference
1. The environment is `Soccer Twos` from ML-Agents Toolkit by Unity Technologies. For more details, check https://github.com/Unity-Technologies/ml-agents/blob/main/docs/ML-Agents-Overview.md.
2. CSCI-GA 3033-090 by Prof. Larrel Pinto. For more details, check https://nyu-robot-learning.github.io/deep-rl-class/.
3. Original paper of MADDPG, by Ryan Lowe et al. 2017. For more details, please check https://arxiv.org/abs/1706.02275.
