import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np 

from model import CentralizedCritic, Actor
from utils import eval_mode


class DDPGAgent:

    def __init__(self, obs_shape, device, num_agents, num_actions, critic_tau, actor_lr, 
                 critic_lr, gamma, num_actions_branch, critic_target_update_frequency, 
                 max_grad_norm, agent_id=None):
        self.actor_lr = actor_lr
        self.critic_lr = critic_lr
        self.gamma = gamma
        self.tau = critic_tau
        self.agent_id = agent_id
        self.device = device
        self.num_agents = num_agents

        self.action_dim = num_actions

        self.critic_input_dim = obs_shape * num_agents
        self.critic_output_dim = (self.action_dim // num_actions_branch) * self.num_agents

        self.actor_input_dim = obs_shape
        
        self.actor_output_dim = num_actions

        self.num_actions_branch = num_actions_branch
        self.critic_target_update_frequency = critic_target_update_frequency
        self.max_grad_norm = max_grad_norm

        self.critic = CentralizedCritic(self.critic_input_dim, self.critic_output_dim).to(self.device)
        self.critic_target = CentralizedCritic(self.critic_input_dim, self.critic_output_dim).to(self.device)
        self.actor = Actor(self.actor_input_dim, self.action_dim).to(self.device)
        self.actor_target = Actor(self.actor_input_dim, self.action_dim).to(self.device)

        for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
            target_param.data.copy_(param.data)
        
        for target_param, param in zip(self.actor_target.parameters(), self.actor.parameters()):
            target_param.data.copy_(param.data)
        
        self.MSELoss = nn.MSELoss()
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=critic_lr)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=actor_lr)

    def get_action(self, state, eps=0.0, eval_flag=False):
        state = state.to(self.device)
        if eval_flag:
            with eval_mode(self.actor):
                action = self.actor.forward(state)
        else:
            action = self.actor.forward(state)
        return self.onehot_from_logits(action, eps)
    
    def onehot_from_logits(self, logits, eps=0.0):
        branch = torch.tensor_split(logits, self.num_actions_branch, dim=0)
        if np.random.rand() > eps:
            actions = torch.stack([b.argmax(axis=0) for b in branch])
        else:
            actions = torch.randint(0, branch[0].shape[0], branch[0].shape)
        return actions
    
    def update(self, indiv_reward_batch, indiv_obs_batch, global_state_batch, global_actions_batch, global_next_state_batch, next_global_actions, logger, step):
        """
        indiv_reward_batch      : only rewards of agent i
        indiv_obs_batch         : only observations of agent i
        global_state_batch      : observations of all agents are concatenated
        global actions_batch    : actions of all agents are concatenated
        global_next_state_batch : observations of all agents are concatenated
        next_global_actions     : actions of all agents are concatenated
        logger     : 
        """
        indiv_reward_batch = torch.tensor(indiv_reward_batch).to(self.device)
        indiv_reward_batch = indiv_reward_batch.view(indiv_reward_batch.size(0), 1).to(self.device) 
        indiv_obs_batch = torch.tensor(np.array(indiv_obs_batch)).to(self.device)
        global_state_batch = torch.tensor(np.array(global_state_batch)).to(self.device)    
        global_actions_batch = torch.stack(global_actions_batch).to(self.device)      
        global_next_state_batch = torch.tensor(np.array(global_next_state_batch)).to(self.device)
        next_global_actions = next_global_actions

        # update critic        
        self.critic_optimizer.zero_grad()
        
        curr_Q = self.critic.forward(global_state_batch, global_actions_batch)
        next_Q = self.critic_target.forward(global_next_state_batch, next_global_actions)
        estimated_Q = indiv_reward_batch + self.gamma * next_Q
        
        critic_loss = self.MSELoss(curr_Q, estimated_Q.detach())
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 0.5)
        self.critic_optimizer.step()

        logger.log('train_critic/loss', critic_loss, step)
        # print(f'train_critic/loss: {critic_loss} at step {step}')

        # update actor
        self.actor_optimizer.zero_grad()

        policy_loss = -self.critic.forward(global_state_batch, global_actions_batch).mean()
        curr_pol_out = self.actor.forward(indiv_obs_batch)
        policy_loss += -(curr_pol_out**2).mean() * 1e-3 
        policy_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm) # max_grad_norm
        self.actor_optimizer.step()
    
    def target_update(self, step):
        for target_param, param in zip(self.actor_target.parameters(), self.actor.parameters()):
            target_param.data.copy_(param.data)
        if step % self.critic_target_update_frequency == 0:
            for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
                target_param.data.copy_(param.data * self.tau + target_param.data * (1.0 - self.tau))        