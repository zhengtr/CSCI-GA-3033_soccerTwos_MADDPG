"""
    The file contains the PPO class to train with.
    NOTE: Original PPO pseudocode can be found here: https://spinningup.openai.com/en/latest/_images/math/e62a8971472597f4b014c2da064f636ffe365ba3.svg
"""

import time

import numpy as np

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.distributions import MultivariateNormal
import os
from os.path import exists

from .base_alg import BasePolicyGradient
from ppo.network import FeedForwardNN

class PPO(BasePolicyGradient):

    def __init__(self,env, obs_shape, num_actions,num_actions_branch,device,timesteps_per_batch,
                 max_timesteps_per_episode,n_updates_per_iteration,lr,gamma,clip,seed, self_play,save_step,num_windows,swap_step,
                 team_change,prob_select_latest_model=0.5,agent_adversary=None):

        super().__init__(env, obs_shape, num_actions,num_actions_branch,device,timesteps_per_batch,
                 max_timesteps_per_episode,n_updates_per_iteration,lr,gamma,clip,seed,self_play,save_step,num_windows,
                 swap_step,team_change,prob_select_latest_model=0.5,agent_adversary=None)
        
        self.critic = FeedForwardNN(obs_shape, 1)
        self.critic_optim = Adam(self.actor.parameters(), lr=lr)

        self.save_path = os.path.join(os.path.dirname(os.path.dirname(os.getcwd())), "ppo_actor.pt")
        self.save_path_critic = os.path.join(os.path.dirname(os.path.dirname(os.getcwd())), "ppo_critic.pt")
        self.selfplay_path = os.path.join(os.getcwd(), "ppo_actor-selfplay.pt")
        self.selfplay_path_critic = os.path.join(os.getcwd(), "ppo_critic-selfplay.pt")
        self.self_play = self_play
        self.save_step = save_step

    def learn(self, total_timesteps, logger):
        """
            Train the actor and critic networks. Here is where the main PPO algorithm resides.

            Parameters:
                total_timesteps - the total number of timesteps to train for

            Return:
                None
        """
        self.logger = logger
        t_so_far = 0 # Timesteps simulated so far
        i_so_far = 0 # Iterations ran so far
        while t_so_far < total_timesteps:
            # We're collecting our batch simulations here
            batch_obs, batch_acts, batch_log_probs, batch_rtgs, batch_lens = self.rollout()

            # Calculate how many timesteps we collected this batch
            t_so_far += np.sum(batch_lens)

            # Increment the number of iterations
            i_so_far += 1

            V, log_probs = self.evaluate(batch_obs, batch_acts, batch_rtgs)
            A_k = batch_rtgs - V.detach()

            # One of the only tricks we use that isn't in the pseudocode. Normalizing advantages
            # isn't theoretically necessary, but in practice it decreases the variance of 
            # our advantages and makes convergence much more stable and faster. I added this because
            # solving some environments was too unstable without it.
            A_k = (A_k - A_k.mean()) / (A_k.std() + 1e-10)

            # This is the loop where we update our network for some n epochs
            for _ in range(self.n_updates_per_iteration):
                V, log_probs = self.evaluate(batch_obs, batch_acts, batch_rtgs)
                probRatios = torch.exp(log_probs - batch_log_probs)

                probs = probRatios * A_k
                clipedProbs = torch.clamp(probRatios, 1 - self.clip, 1 + self.clip) * A_k

                actor_loss = -torch.min(probs, clipedProbs).mean()
                critic_loss = nn.MSELoss()(V, batch_rtgs)

                # Calculate gradients and perform backward propagation for actor network
                self.actor_optim.zero_grad()
                actor_loss.backward(retain_graph=True)
                self.actor_optim.step()

                # Calculate gradients and perform backward propagation for critic network
                self.critic_optim.zero_grad()
                critic_loss.backward()
                self.critic_optim.step()
            self.logging()


    def evaluate(self, batch_obs, batch_acts, batch_rtgs):
        """
            Estimate the values of each observation, and the log probs of
            each action in the most recent batch with the most recent
            iteration of the actor network. Should be called from learn.

            Parameters:
                batch_obs - the observations from the most recently collected batch as a tensor.
                            Shape: (number of timesteps in batch, dimension of observation)
                batch_acts - the actions from the most recently collected batch as a tensor.
                            Shape: (number of timesteps in batch, dimension of action)
                batch_rtgs - the rewards-to-go calculated in the most recently collected
                                batch as a tensor. Shape: (number of timesteps in batch)
        """

        V = self.critic(batch_obs).squeeze()

        # Calculate the log probabilities of batch actions using most recent actor network.
        # This segment of code is similar to that in get_action()
        mean = self.actor(batch_obs)
        dist = MultivariateNormal(mean, self.cov_mat)
        log_probs = dist.log_prob(batch_acts)

        # Return the value vector V of each observation in the batch
        # and log probabilities log_probs of each action in the batch
        return V, log_probs

    def save_checkpoint(self, self_play=False, window=0):
        if self_play:
            torch.save(self.actor.state_dict(), str(window)+"_"+self.selfplay_path)
            torch.save(self.actor.state_dict(), str(window)+"_"+self.selfplay_path_critic)
        torch.save(self.actor.state_dict(), self.save_path)
        torch.save(self.critic.state_dict(), self.save_path_critic)

    def selfplay_load_agent(self, window):
        path = str(window)+"_"+self.selfplay_path
        path_critic = str(window) + "_" + self.selfplay_path_critic
        if exists(path):
            self.actor.load_state_dict(torch.load(path))
            self.critic.load_state_dict(torch.load(path_critic))