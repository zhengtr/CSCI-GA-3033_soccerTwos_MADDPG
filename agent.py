import torch
import numpy as np 
import torch.optim as optim
from model import Actor
from utils import eval_mode

class DDPGAgent:

    def __init__(self, input_dim, output_dim, num_actions_branch, device, lr, gamma, critic, 
                 max_grad_norm, loss_fn, agent_id=None):

        self.agent_id = agent_id
        self.device = device
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.lr = lr
        self.gamma = gamma
        self.num_actions_branch = num_actions_branch

        self.actor = Actor(self.input_dim, self.output_dim).to(self.device)
        self.actor_target = Actor(self.input_dim, self.output_dim).to(self.device)

        self.critic = critic
        self.MSELoss = loss_fn
        for target_param, param in zip(self.actor_target.parameters(), self.actor.parameters()):
            target_param.data.copy_(param.data)

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr)
        self.max_grad_norm = max_grad_norm

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
    
    def update(self, curr_Q, next_Q, indiv_reward_batch, indiv_obs_batch, global_state_batch, global_actions_batch, critic_optimizer, logger, step):
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

        # update critic        
        
        estimated_Q = indiv_reward_batch + self.gamma * next_Q
        
        critic_loss = self.MSELoss(curr_Q, estimated_Q.detach())
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
        critic_optimizer.step()

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
    
    def target_update(self):
        for target_param, param in zip(self.actor_target.parameters(), self.actor.parameters()):
            target_param.data.copy_(param.data)
