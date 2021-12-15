import os
from os.path import exists

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

from agent import DDPGAgent
from replay_buffer import MultiAgentReplayBuffer
from model import CentralizedCritic, Actor
from agent import DDPGAgent

class MADDPG:

    def __init__(self, obs_shape, num_actions, num_actions_branch, device, actor_lr, critic_lr,
                 critic_tau, gamma, eval_eps, num_agents, critic_target_update_frequency, max_grad_norm, 
                 replay_buffer_capacity, batch_size, logger=None):
       
        self.num_agents = num_agents
        self.device = device
        self.logger = logger

        self.actor_lr = actor_lr
        self.critic_lr = critic_lr
        self.gamma = gamma
        self.tau = critic_tau
        self.max_grad_norm = max_grad_norm
        self.critic_target_update_frequency = critic_target_update_frequency
        self.batch_size = batch_size
        self.eval_eps = eval_eps

        self.obs_shape = obs_shape
        self.num_actions = num_actions
        self.num_actions_branch = num_actions_branch

        self.replay_buffer_capacity = replay_buffer_capacity
        self.replay_buffer = MultiAgentReplayBuffer(self.num_agents, self.replay_buffer_capacity)

        self.critic_input_dim = obs_shape * num_agents
        self.critic_output_dim = (num_actions // num_actions_branch) * self.num_agents
        self.actor_input_dim = obs_shape
        self.actor_output_dim = num_actions

        self.critic = CentralizedCritic(self.critic_input_dim, self.critic_output_dim).to(self.device)
        self.critic_target = CentralizedCritic(self.critic_input_dim, self.critic_output_dim).to(self.device)
        for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
            target_param.data.copy_(param.data)
        
        self.MSELoss = nn.MSELoss()
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=critic_lr)

        self.agents = [
            DDPGAgent(self.actor_input_dim, self.actor_output_dim, self.num_actions_branch, 
                      self.device, self.actor_lr, self.gamma, self.critic, max_grad_norm, 
                      self.MSELoss, i
            )
            for i in range(self.num_agents)
        ]
        self.agent_save_path = os.path.join(os.path.dirname(os.path.dirname(os.getcwd())), "ddpg_agent.pt")
        self.critic_save_path = os.path.join(os.path.dirname(os.path.dirname(os.getcwd())), "MADDPG_critic.pt")

        self.step = 0

    def get_actions(self, states, eps=0.0, eval_flag=False):
        actions = []
        for i in range(self.num_agents):
            action = self.agents[i].get_action(states[i], eps=eps, eval_flag=eval_flag).cpu().detach().numpy().copy()
            actions.append(action)
        return actions

    def update_critic(self, global_state_batch, global_actions_batch, global_next_state_batch, next_global_actions):
        global_state_batch = torch.tensor(np.array(global_state_batch)).to(self.device)    
        global_actions_batch = torch.stack(global_actions_batch).to(self.device)      
        global_next_state_batch = torch.tensor(np.array(global_next_state_batch)).to(self.device)
        self.critic_optimizer.zero_grad()
        curr_Q = self.critic.forward(global_state_batch, global_actions_batch)
        next_Q = self.critic_target.forward(global_next_state_batch, next_global_actions)
        return curr_Q, next_Q

    def train(self):
        obs_batch, indiv_action_batch, indiv_reward_batch, next_obs_batch, \
            global_state_batch, global_actions_batch, global_next_state_batch, done_batch = self.replay_buffer.sample(self.batch_size)

        
        train_batch_reward = 0.0
        for i in range(self.num_agents):
            obs_batch_i = obs_batch[i]
            indiv_action_batch_i = indiv_action_batch[i]
            indiv_reward_batch_i = indiv_reward_batch[i]
            next_obs_batch_i = next_obs_batch[i]

            train_batch_reward += np.sum(indiv_reward_batch_i)

            next_global_actions = []
            for agent in self.agents:
                next_obs_batch_i = torch.FloatTensor(np.array(next_obs_batch_i))
                indiv_next_action = agent.actor.forward(next_obs_batch_i.to(self.device))
                indiv_next_action = [agent.onehot_from_logits(indiv_next_action_j) for indiv_next_action_j in indiv_next_action]
                indiv_next_action = torch.stack(indiv_next_action)
                next_global_actions.append(indiv_next_action)

            a = torch.stack([next_actions_i for next_actions_i in next_global_actions])
            next_global_actions = a.permute(1,0,2).reshape(self.batch_size, -1).contiguous()


            # using global info get critic info
            curr_Q, next_Q = self.update_critic(global_state_batch, global_actions_batch, global_next_state_batch, next_global_actions)

            self.agents[i].update(curr_Q, next_Q, indiv_reward_batch_i, obs_batch_i, global_state_batch, global_actions_batch, self.critic_optimizer, self.logger, self.step)
            self.agents[i].target_update()

        if self.step % self.critic_target_update_frequency == 0:
            for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
                target_param.data.copy_(param.data * self.tau + target_param.data * (1.0 - self.tau))        
        self.logger.log('train/batch_reward', train_batch_reward.mean(), self.step)

    def on_begin(self):
        if exists(self.agent_save_path):
            print('Saved agent model found\n')
            for i in range(self.num_agents):
                self.agents[i].actor_target.load_state_dict(torch.load(self.agent_save_path))
                self.agents[i].actor_target.to(self.device)
        if exists(self.critic_save_path):
            print('Saved critic model found\n')
            self.critic.load_state_dict(torch.load(self.critic_save_path))
            self.critic.to(self.device)
            for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
                target_param.data.copy_(param.data)
            self.critic_target.to(self.device)

    def on_exit(self):
        print('Trained models saved\n')
        torch.save(self.agents[0].actor_target.state_dict(), self.agent_save_path)
        torch.save(self.critic.state_dict(), self.critic_save_path)
