"""
    The file contains the basics of any policy gradient algorithm class to train with.
"""

import time
from mlagents_envs.side_channel.engine_configuration_channel import EngineConfigurationChannel
import numpy as np
from mlagents_envs.registry import default_registry
import torch
from torch.optim import Adam
from torch.distributions import MultivariateNormal
from mlagents_envs.environment import ActionTuple
from ppo.network import FeedForwardNN

class BasePolicyGradient:
    """
        This is the base policy gradient class we will use as our model in main.py
    """
    def __init__(self, env, obs_shape, num_actions,num_actions_branch,device,timesteps_per_batch,
                 max_timesteps_per_episode,n_updates_per_iteration,lr,gamma,clip,seed,self_play,save_step,num_windows,
                 swap_step,team_change,prob_select_latest_model=0.5,agent_adversary=None):

         # Initialize actor and critic networks
        self.actor = FeedForwardNN(obs_shape, num_actions*num_actions_branch)

        # Initialize optimizers for actor and critic
        self.actor_optim = Adam(self.actor.parameters(), lr=lr)

        # Initialize the covariance matrix used to query the actor for actions
        self.cov_var = torch.full(size=(num_actions*num_actions_branch,), fill_value=0.5)
        self.cov_mat = torch.diag(self.cov_var)

        self.timesteps_per_batch = timesteps_per_batch
        self.max_timesteps_per_episode = max_timesteps_per_episode
        self.gamma = gamma
        self.clip = clip
        self.n_updates_per_iteration = n_updates_per_iteration

        channel = EngineConfigurationChannel()
        channel.set_configuration_parameters(time_scale=0.0)
        self.env = default_registry[env].make(seed=seed, worker_id=4, side_channels=[channel], no_graphics=True)
        self.env.reset()
        self.behavior_name = list(self.env.behavior_specs)[0]

        self.self_play = self_play
        if self.self_play:
            self.behavior_name_adversary = list(self.env.behavior_specs)[1]
            # self.group_adversary = self.field_detative(self.behavior_name_adversary)

            self.save_step = save_step
            self.num_windows = num_windows
            self.window = -1
            self.window_map = list(range(1,11))
            self.swap_step = swap_step
            self.prob_select_latest_model = prob_select_latest_model
            self.agent_adversary = agent_adversary
            self.team_change = team_change

    def learn(self, total_timesteps, logger):
        """
            Train the networks. Here is where the main algorithms resides.

            Parameters:
                total_timesteps - the total number of timesteps to train for
        """
        raise NotImplementedError("Implement this in the policy subclass")

    def window_select(self):
        if self.window < 1:
            return self.window

        if np.random.rand() < self.prob_select_latest_model:
            return np.random.uniform(0, self.window-1)
        else:
            return self.window

    def rollout(self):
        """
            This is where we collect the batch of data from simulation. 
            Since this is an on-policy algorithm, we'll need to collect a fresh batch
            of data each time we iterate the actor/critic networks.

            Parameters:
                None

            Return:
                batch_obs - the observations collected this batch. Shape: (number of timesteps, dimension of observation)
                batch_acts - the actions collected this batch. Shape: (number of timesteps, dimension of action)
                batch_log_probs - the log probabilities of each action taken this batch. Shape: (number of timesteps)
                batch_rtgs - the Rewards-To-Go of each timestep in this batch. Shape: (number of timesteps)
                batch_lens - the lengths of each episode this batch. Shape: (number of episodes)
        """
        # Batch data. For more details, check function header.
        batch_obs = []
        batch_acts = []
        batch_log_probs = []
        batch_rews = []
        batch_rtgs = []
        batch_lens = []

        # Episodic data. Keeps track of rewards per episode, will get cleared
        # upon each new episode
        ep_rews = []

        t = 0 # Keeps track of how many timesteps we've run so far this batch

        # Keep simulating until we've run more than or equal to specified timesteps per batch
        while t < self.timesteps_per_batch:
            ep_rews = [] # rewards collected per episode

            # Reset the environment. sNote that obs is short for observation. 
            self.env.reset()
            decision_steps, terminal_steps = self.env.get_steps(self.behavior_name)

            # Run an episode for a maximum of max_timesteps_per_episode timesteps
            for ep_t in range(self.max_timesteps_per_episode):

                if self.self_play and t > 0:
                    if t % self.save_step == 0:
                        self.window += 1
                        win = self.window
                        if self.window >= self.num_windows:
                            win = self.window_map[0]
                            self.window -= 1
                        self.save_checkpoint(self_play=True, window=win)

                    if t % self.swap_step == 0:
                        win = self.window_select()
                        self.agent_adversary.selfplay_load_agent(self.window_map[win])


                t += 1 # Increment timesteps ran this batch so far

                # Track observations in this batch
                obs = torch.FloatTensor(np.array([np.concatenate(decision_steps[i].obs) for i in decision_steps.agent_id]))
                batch_obs.append(obs)

                # Calculate action and make a step in the env. 
                # Note that rew is short for reward.
                actions, log_prob = self.get_action(obs)
                action = ActionTuple()
                action.add_discrete(actions)
                self.env.set_actions(self.behavior_name, action)
                self.env.step()
                next_decision_steps, next_terminal_steps = self.env.get_steps(self.behavior_name)
                rew = 0
                for agent_i in next_terminal_steps.agent_id:
                    if agent_i in decision_steps.agent_id:
                        reward = next_terminal_steps[agent_i].reward
                        pre_obs = decision_steps[agent_i].obs
                        index = decision_steps.agent_id_to_index[agent_i]
                        act = torch.FloatTensor(action.discrete[index])
                        rew += reward

                # adding to replay buffer
                for agent_i in next_decision_steps.agent_id:
                    reward = next_decision_steps[agent_i].reward
                    index = next_decision_steps.agent_id_to_index[agent_i]
                    act = torch.FloatTensor(action.discrete[index])

                    rew += reward

                # Track recent reward, action, and action log probability
                ep_rews.append(rew)
                batch_acts.append(action)
                batch_log_probs.append(log_prob)

                # If the environment tells us the episode is terminated, break
                if not next_decision_steps or next_terminal_steps:
                    break

            # Track episodic lengths and rewards
            batch_lens.append(ep_t + 1)
            batch_rews.append(ep_rews)

        # Reshape data as tensors in the shape specified in function description, before returning
        batch_obs = torch.tensor(batch_obs, dtype=torch.float)
        batch_acts = torch.tensor(batch_acts, dtype=torch.float)
        batch_log_probs = torch.tensor(batch_log_probs, dtype=torch.float).flatten()
        batch_rtgs = self.compute_rtgs(batch_rews)

        self.logger.log('train/episode_reward', ep_rews, t)

        return batch_obs, batch_acts, batch_log_probs, batch_rtgs, batch_lens

    def compute_rtgs(self, batch_rews):
        """
            Compute the Reward-To-Go of each timestep in a batch given the rewards.
            This is just the rewards normalized by the reward discount gamma and added to the past rewards.

            Parameters:
                batch_rews - the rewards in a batch, Shape: (number of episodes, number of timesteps per episode)

            Return:
                batch_rtgs - the rewards to go, Shape: (number of timesteps in batch)
        """
        # The rewards-to-go (rtg) per episode per batch to return.
        # The shape will be (num episodes per batch, num timesteps per episode)
        batch_rtgs = []

        # Iterate through each episode
        for ep_rews in reversed(batch_rews):

            discounted_reward = 0 # The discounted reward so far

            # Iterate through all rewards in the episode. We go backwards for smoother calculation of each
            # discounted return (think about why it would be harder starting from the beginning)
            for rew in reversed(ep_rews):
                discounted_reward = rew + discounted_reward * self.gamma
                batch_rtgs.insert(0, discounted_reward)

        # Convert the rewards-to-go into a tensor
        batch_rtgs = torch.tensor(batch_rtgs, dtype=torch.float,)

        return batch_rtgs

    def get_action(self, obs):
        """
            Queries an action from the actor network, should be called from rollout.

            Parameters:
                obs - the observation at the current timestep

            Return:
                action - the action to take, as a numpy array
                log_prob - the log probability of the selected action in the distribution
        """
        # Query the actor network for a mean action
        mean = self.actor(obs)

        # Create a distribution with the mean action and std from the covariance matrix above.
        # For more information on how this distribution works, check out Andrew Ng's lecture on it:
        # https://www.youtube.com/watch?v=JjB58InuTqM
        dist = MultivariateNormal(mean, self.cov_mat)

        # Sample an action from the distribution
        action = dist.sample()

        # Calculate the log probability for that action
        log_prob = dist.log_prob(action)

        # Return the sampled action and the log probability of that action in our distribution
        return action.detach().numpy(), log_prob.detach()

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
        raise NotImplementedError("Implement this method in the policy gradient algorithm")



