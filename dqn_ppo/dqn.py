import hydra
import os
import numpy as np
import torch
import torch.nn as nn
import torch.autograd as autograd
import torch.nn.functional as F
from replay_buffer import PrioritizedReplayBuffer
import utils
from os.path import exists

class Critic(nn.Module):
    def __init__(self, input_dim, num_actions, dueling, device):
        super(Critic, self).__init__()
        self.input_dim = input_dim
        self.num_actions = num_actions
        self.device = device

        self.dueling = dueling
        if dueling:
            self.V = nn.Sequential(nn.Linear(self.input_dim, 64),
                                   nn.ReLU(),
                                   nn.Linear(64, 1))
            self.A = nn.Sequential(nn.Linear(self.input_dim, 64),
                                   nn.ReLU(),
                                   nn.Linear(64, self.num_actions))
        else:
            self.fc = nn.Sequential(nn.Linear(self.input_dim, 512),
                                    nn.ReLU(),
                                    nn.Linear(512, 512),
                                    nn.ReLU(),
                                    nn.Linear(512, self.num_actions)
                                    )


    def forward(self, obs):
        obs = obs.to(self.device)
        if self.dueling:
            v = self.V(obs)
            a = self.A(obs)
            b = a.mean()
            q = v + (a - a.mean())
        else:
            q = self.fc(obs)
        return q


class DRQLAgent(object):
    """Data regularized Q-learning: Deep Q-learning."""
    def __init__(self, obs_shape, num_actions, num_actions_branch, device, critic_cfg,
                 discount, lr, beta_1, beta_2, weight_decay, adam_eps,
                 max_grad_norm, critic_tau, critic_target_update_frequency,
                 batch_size, multistep_return, eval_eps, prioritized_replay_beta0,double_q,
                 prioritized_replay_beta_steps):

        self.device = device
        self.discount = discount
        self.critic_tau = critic_tau
        self.num_actions = num_actions
        self.num_actions_branch = num_actions_branch
        self.critic_target_update_frequency = critic_target_update_frequency
        self.batch_size = batch_size
        self.eval_eps = eval_eps
        self.max_grad_norm = max_grad_norm
        self.multistep_return = multistep_return
        self.double_q = False
        self.double_q = double_q
        assert prioritized_replay_beta0 <= 1.0
        self.prioritized_replay_beta0 = prioritized_replay_beta0
        self.prioritized_replay_beta_steps = prioritized_replay_beta_steps
        self.eps = 0

        self.critic = hydra.utils.instantiate(critic_cfg).to(self.device)
        self.critic_target = hydra.utils.instantiate(critic_cfg).to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())

        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(),
                                                 lr=lr,
                                                 betas=(beta_1, beta_2),
                                                 weight_decay=weight_decay,
                                                 eps=adam_eps)

        self.train()
        self.critic_target.train()
        self.save_path = os.path.join(os.path.dirname(os.path.dirname(os.getcwd())), "dqnModel.pt")
        self.selfplay_path = os.path.join(os.getcwd(), "dqnModel-selfplay.pt")

    def q_extra_max_action(self, q):
        q_np = q.cpu().detach().numpy().copy()
        branch = np.array_split(q_np, self.num_actions_branch, axis=1)
        actions = [b.argmax(axis=1) for b in branch]
        action = [[b[agent_i] for b in actions] for agent_i in range(q.shape[0])]
        return np.array(action)

    def q_extra_max(self, q):
        q_np = q.cpu().detach().numpy().copy()
        branch = np.array_split(q_np, self.num_actions_branch, axis=1)
        actions = [b.max(axis=1) for b in branch]
        action = [[b[agent_i] for b in actions] for agent_i in range(q.shape[0])]
        return np.array(action)

    def train(self, training=True):
        self.training = training
        self.critic.train(training)

    def act(self, obs):
        with torch.no_grad():
            obs = obs.to(self.device)
            obs = obs.contiguous()
            q = self.critic(obs)
            # q_np = q.cpu().detach().numpy().copy()
            # branch = np.array_split(q_np, self.num_actions_branch, axis=1)
            # actions = [b.argmax(axis=1) for b in branch]
            # action = [[b[agent_i] for b in actions] for agent_i in range(q.shape[0])]
            # make sure action is a [16,3] nparray before return
        return self.q_extra_max_action(q)

    def update_critic(self, obs, action, reward, next_obs, not_done, weights,
                      logger, step):
        with torch.no_grad():
            discount = self.discount**self.multistep_return
            if self.double_q:
                # TODO double Q learning
                # Find the target Q value based on the critic
                # and the critic target networks to find the right
                # value of target_Q
                current_temp_q = self.critic(next_obs)
                current_max_action = self.q_extra_max_action(current_temp_q)
                actions_offset = [[act[i] + 3 * i for i in range(self.num_actions_branch)] for act in current_max_action]
                next_Q = self.critic_target(next_obs)
                next_Q = next_Q.gather(1, torch.tensor(actions_offset).to(self.device))
                target_Q = reward + (not_done * discount * next_Q)
                # End TODO
            else:
                next_Q = self.critic_target(next_obs)
                actions = self.q_extra_max_action(next_Q)
                actions_offset = [[act[i]+3*i for i in range(self.num_actions_branch)] for act in actions]
                next_Q = next_Q.gather(1, torch.tensor(actions_offset).to(self.device))
                target_Q = reward + (not_done * discount * next_Q)

        # get current Q estimates
        current_Q = self.critic(obs)
        actions = self.q_extra_max_action(current_Q)
        actions_offset = [[act[i] + 3 * i for i in range(self.num_actions_branch)] for act in actions]
        current_Q = current_Q.gather(1, torch.tensor(actions_offset).to(self.device))

        td_errors = current_Q - target_Q
        critic_losses = F.smooth_l1_loss(current_Q, target_Q, reduction='none')
        if weights is not None:
            critic_losses *= weights

        critic_loss = critic_losses.mean()

        logger.log('train_critic/loss', critic_loss, step)

        # Optimize the critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        if self.max_grad_norm > 0.0:
            nn.utils.clip_grad_norm_(self.critic.parameters(),
                                     self.max_grad_norm)
        self.critic_optimizer.step()

        # self.critic.log(logger, step)

        return td_errors.squeeze(dim=1).cpu().detach().numpy()

    def update(self, replay_buffer, logger, step):

        prioritized_replay = type(replay_buffer) == PrioritizedReplayBuffer
        # prioritized_replay = True
        if prioritized_replay:
            fraction = min(step / self.prioritized_replay_beta_steps, 1.0)
            beta = self.prioritized_replay_beta0 + fraction * (
                1.0 - self.prioritized_replay_beta0)
            obs, action, reward, next_obs, not_done, weights, idxs = replay_buffer.sample_multistep(
                self.batch_size, beta, self.discount, self.multistep_return)
        else:
            obs, action, reward, next_obs, not_done = replay_buffer.sample_multistep(
                self.batch_size, self.discount, self.multistep_return)
            weights = None

        logger.log('train/batch_reward', reward.mean(), step)

        td_errors = self.update_critic(obs, action, reward, next_obs, not_done,
                                       weights, logger, step)
        td_errors = td_errors.sum(axis=1)

        if prioritized_replay:
            prios = np.absolute(td_errors) + 0.1
            replay_buffer.update_priorities(idxs, prios)

        if step % self.critic_target_update_frequency == 0:
            utils.soft_update_params(self.critic, self.critic_target,
                                     self.critic_tau)

    def on_begin(self):
        if exists(self.save_path):
            # model = Critic()
            self.critic.load_state_dict(torch.load(self.save_path))
            self.critic.to(self.device)
            # self.critic = model
            self.critic_target.load_state_dict(self.critic.state_dict())

    def save_checkpoint(self, self_play=False, window=0):
        if self_play:
            torch.save(self.critic.state_dict(), self.selfplay_path+"_"+str(window))
        torch.save(self.critic.state_dict(), self.save_path)

    def selfplay_load_agent(self, window):
        path = self.selfplay_path+"_"+str(window)
        if exists(path):
            self.critic.load_state_dict(torch.load(path))
            self.critic.to(self.device)
            self.critic_target.load_state_dict(self.critic.state_dict())

