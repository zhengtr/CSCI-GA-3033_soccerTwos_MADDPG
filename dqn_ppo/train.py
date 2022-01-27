import hydra
import numpy
import torch
import os
from logger import Logger
import wandb
import numpy as np
from mlagents_envs.side_channel.engine_configuration_channel import EngineConfigurationChannel
from mlagents_envs.registry import default_registry
from replay_buffer import ReplayBuffer, PrioritizedReplayBuffer
from mlagents_envs.environment import ActionTuple
import utils

# sns.set()
class Workspace(object):
    def __init__(self, cfg):
        self.work_dir = os.getcwd()
        print(f'workspace: {self.work_dir}')

        self.cfg = cfg
        self.logger = Logger(self.work_dir,
                             save_tb=cfg.log_save_tb,
                             log_frequency=cfg.log_frequency_step,
                             agent=cfg.agent.name)

        self.device = torch.device(cfg.device)
        self.seed = cfg.seed
        utils.set_seed_everywhere(cfg.seed)
        self.checkPoint_freq = cfg.save_frequency

        # env set up
        channel = EngineConfigurationChannel()
        channel.set_configuration_parameters(time_scale=0.0)
        self.env = default_registry[cfg.env].make(seed=self.seed, worker_id=cfg.work_id, side_channels=[channel], no_graphics=True)
        self.env.reset()
        self.behavior_name = list(self.env.behavior_specs)[0]

        eval_channel = EngineConfigurationChannel()
        eval_channel.set_configuration_parameters(time_scale=0.0)
        self.eval_env = default_registry[cfg.env].make(seed=self.seed+1, worker_id=cfg.work_id_ev, side_channels=[eval_channel], no_graphics=True)
        self.eval_env.reset()
        self.eval_behavior_name = list(self.eval_env.behavior_specs)[0]

        self.spec = self.env.behavior_specs[self.behavior_name]
        cfg.agent.params.obs_shape = sum(i.shape[0] for i in self.spec.observation_specs)
        if self.spec.action_spec.is_continuous():
            cfg.agent.params.num_actions = self.spec.action_spec.continuous_size
        elif self.spec.action_spec.is_discrete():
            cfg.agent.params.num_actions = sum(list(self.spec.action_spec.discrete_branches))
            cfg.agent.params.num_actions_branch = self.spec.action_spec.discrete_size

        self.ppo = cfg.ppo
        if self.ppo:
            cfg.agent_ppo.params.num_actions = cfg.agent.params.num_actions
            cfg.agent_ppo.params.num_actions_branch = cfg.agent.params.num_actions_branch
            cfg.agent_ppo.params.obs_shape = cfg.agent.params.obs_shape
            self.agent = hydra.utils.instantiate(cfg.agent_ppo)
            self.agent.learn(cfg.num_train_steps, self.logger)
        
        
        self.group = self.field_detative(self.behavior_name)

        self.self_play = cfg.self_play
        if self.self_play:
            self.behavior_name_adversary = list(self.env.behavior_specs)[1]
            # self.group_adversary = self.field_detative(self.behavior_name_adversary)

            self.save_step = cfg.save_step
            self.num_windows = cfg.num_windows
            self.window = -1
            self.window_map = list(range(1,11))
            self.swap_step = cfg.swap_step
            self.team_change = cfg.team_change
            self.prob_select_latest_model = cfg.prob_select_latest_model
            self.agent_adversary = hydra.utils.instantiate(cfg.agent)

        self.step = 0
        self.average_episode_reward = 0

        self.agent = hydra.utils.instantiate(cfg.agent)
        # self.agent.on_begin()

        if cfg.prioritized_replay:
            self.replay_buffer = PrioritizedReplayBuffer(cfg.agent.params.obs_shape, cfg.replay_buffer_capacity,
                                                            cfg.prioritized_replay_alpha, self.device)
        else:
            self.replay_buffer = ReplayBuffer(cfg.agent.params.obs_shape, cfg.replay_buffer_capacity, self.device)

        # wandb.init(project="Soccer2", group=self.env, name="Seed-" + str(self.seed), entity="dariaxu")
        # wandb.init(project="Soccer2-dqp", entity="dariaxu")

    def field_detative(self, behavior_name):
        decision_steps, terminal_steps = self.env.get_steps(behavior_name)
        action = self.spec.action_spec.random_action(len(decision_steps))
        num_agents = decision_steps.agent_id.size
        list_group = [[] for i in range(int(num_agents/2))]
        list_checked = []
        num_group = 0

        for i in range(num_agents):
            if any(i in k for k in list_checked):
                continue
            action.discrete[:] = [0,0,0]
            action.discrete[i] = [2,2,2]

            for n in range(10):
                self.env.set_actions(behavior_name, action)
                self.env.step()
                next_decision_steps, next_terminal_steps = self.env.get_steps(behavior_name)

            diffs =[]
            for j in decision_steps.agent_id:
                diffs.append(np.sum(decision_steps[j].obs[0]-next_decision_steps[j].obs[0]))

            b, index = np.unique(diffs, return_index=True)
            grouped = []
            for first in index:
                if first == len(diffs)-1:
                    grouped.append(first)
                    continue
                if not diffs[first] in diffs[first+1:]:
                    grouped.append(first)

            list_checked.append(grouped)
            list_group[num_group] = grouped
            num_group += 1
            self.env.reset()

        return list_group

    def window_select(self):
        if self.window < 1:
            return self.window

        if np.random.rand() < self.prob_select_latest_model:
            return np.random.uniform(0, self.window-1)
        else:
            return self.window

    def evaluate(self):
        self.average_episode_reward = 0
        eval_step = 0
        num_eval_episodes = 0
        while eval_step < self.cfg.num_eval_steps:
            self.eval_env.reset()
            decision_steps, terminal_steps = self.eval_env.get_steps(self.eval_behavior_name)
            num_agents = len(decision_steps)

            episode_reward = 0
            episode_step = 0
            while decision_steps or not terminal_steps:  # no goal yet
                if np.random.rand() < self.agent.eval_eps:
                    action = self.spec.action_spec.random_action(len(decision_steps))
                else:
                    obs = torch.FloatTensor(np.array([np.concatenate(decision_steps[i].obs) for i in decision_steps.agent_id]))
                    with utils.eval_mode(self.agent):
                        action_np = self.agent.act(obs)
                        action = ActionTuple()
                        action.add_discrete(action_np)

                self.eval_env.set_actions(self.eval_behavior_name, action)
                self.eval_env.step()
                next_decision_steps, next_terminal_steps = self.eval_env.get_steps(self.eval_behavior_name)

                for agent_i in next_decision_steps.agent_id:
                    episode_reward += next_decision_steps[agent_i].reward

                for agent_i in next_terminal_steps.agent_id:
                    episode_reward += next_terminal_steps[agent_i].reward

                episode_step += 1
                eval_step += 1

                decision_steps = next_decision_steps
                terminal_steps = next_terminal_steps

            self.average_episode_reward += episode_reward/num_agents
            num_eval_episodes += 1

        if self.average_episode_reward > 0:
            self.average_episode_reward /= num_eval_episodes
        self.logger.log('eval/episode_reward', self.average_episode_reward,
                        self.step)
        self.logger.dump(self.step, ty='eval')

    def run(self):
        episode, episode_reward, episode_step = 0, 0, 1
        self.average_episode_reward = 0
        decision_steps, terminal_steps = self.env.get_steps(self.behavior_name)
        num_agents = len(decision_steps)

        while self.step < self.cfg.num_train_steps:
            if self.step > 0 and self.step % self.checkPoint_freq == 0:
                self.agent.save_checkpoint()

            if self.self_play and self.step > 0:
                if self.step % self.save_step == 0:
                    self.window += 1
                    win = self.window
                    if self.window >= self.num_windows:
                        win = self.window_map[0]
                        self.window -= 1
                    self.agent.save_checkpoint(self_play=True, window=win)

                if self.step % self.swap_step == 0:
                    win = int(self.window_select())
                    self.agent_adversary.selfplay_load_agent(self.window_map[win])

                if self.step % self.team_change == 0:
                    temp = self.agent
                    self.agent = self.agent_adversary
                    self.agent_adversary = temp

            if not decision_steps or terminal_steps:
                if self.step > 0:
                    episode_reward = episode_reward/num_agents
                    self.logger.log('train/episode_reward', episode_reward,
                                    self.step)
                    self.logger.log('train/episode', episode, self.step)
                    self.logger.dump(
                        self.step,
                        save=(self.step > self.cfg.start_training_steps),
                        ty='train')

                    # wandb.log(
                    #     {"Train Episodic Return": episode_reward, "Average Episodic Return": self.average_episode_reward}, step=self.step)

                self.env.reset()
                decision_steps, terminal_steps = self.env.get_steps(self.behavior_name)
                num_agents = len(decision_steps)
                episode_reward = 0
                episode_step = 0
                episode += 1

            # evaluate agent periodically
            if self.step > 0 and self.step % self.cfg.eval_frequency == 0:
                self.logger.log('eval/episode', episode, self.step)
                self.evaluate()

            steps_left = self.cfg.num_exploration_steps + self.cfg.start_training_steps - self.step
            bonus = (1.0 - self.cfg.min_eps) * steps_left / self.cfg.num_exploration_steps
            bonus = np.clip(bonus, 0., 1. - self.cfg.min_eps)
            self.agent.eps = self.cfg.min_eps + bonus

            self.logger.log('train/eps', self.agent.eps, self.step)
            obs = torch.FloatTensor(np.array([np.concatenate(decision_steps[i].obs) for i in decision_steps.agent_id]))
            # sample action for data collection
            if np.random.rand() < self.agent.eps:
                action = self.spec.action_spec.random_action(len(decision_steps))
            else:
                with utils.eval_mode(self.agent):
                    action_np = self.agent.act(obs)
                    action = ActionTuple()
                    action.add_discrete(action_np)

            # run training update
            if self.step >= self.cfg.start_training_steps:
                for _ in range(self.cfg.num_train_iters):
                    self.agent.update(self.replay_buffer, self.logger, self.step)

            self.env.set_actions(self.behavior_name, action)
            if self.self_play and self.step >= self.swap_step:
                decision_steps_ad, terminal_steps_ad = self.env.get_steps(self.behavior_name_adversary)
                obs_ad = torch.FloatTensor(
                    np.array([np.concatenate(decision_steps_ad[i].obs) for i in decision_steps_ad.agent_id]))
                action_adversary = self.agent_adversary.act(obs_ad)
                action_ad = ActionTuple()
                action_ad.add_discrete(action_adversary)
                self.env.set_actions(self.behavior_name_adversary, action_ad)

            self.env.step()
            next_decision_steps, next_terminal_steps = self.env.get_steps(self.behavior_name)

            for agent_i in next_terminal_steps.agent_id:
                if agent_i in decision_steps.agent_id:
                    reward = next_terminal_steps[agent_i].reward
                    pre_obs = decision_steps[agent_i].obs
                    index = decision_steps.agent_id_to_index[agent_i]
                    act = torch.FloatTensor(action.discrete[index])
                    self.replay_buffer.add(np.concatenate(pre_obs), act,
                                           reward, np.concatenate(next_terminal_steps[agent_i].obs), float(True))
                    episode_reward += reward

            # adding to replay buffer
            for agent_i in next_decision_steps.agent_id:
                reward = next_decision_steps[agent_i].reward
                index = next_decision_steps.agent_id_to_index[agent_i]
                act = torch.FloatTensor(action.discrete[index])
                self.replay_buffer.add(np.concatenate(decision_steps[agent_i].obs), act,
                                       reward, np.concatenate(next_decision_steps[agent_i].obs), float(False))
                episode_reward += reward

            decision_steps = next_decision_steps
            terminal_steps = next_terminal_steps

            episode_step += 1
            self.step += 1

        self.env.close()
        self.eval_env.close()
        # self.agent.save_checkpoint()


@hydra.main(config_path='config.yaml')
def main(cfg):
    # input = torch.cuda.is_available()
    from train import Workspace as W
    workspace = W(cfg)
    # for i in range(10):
    #     workspace.run()

    workspace.run()

if __name__ == '__main__':
    main()