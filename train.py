import hydra
import torch
import os
from logger import Logger
import wandb
import numpy as np
from mlagents_envs.side_channel.engine_configuration_channel import EngineConfigurationChannel
from mlagents_envs.registry import default_registry
from replay_buffer import MultiAgentReplayBuffer, ReplayBuffer
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
                             agent=cfg.model.name)

        self.seed = cfg.seed
        utils.set_seed_everywhere(cfg.seed)

        # env set up
        channel = EngineConfigurationChannel()
        channel.set_configuration_parameters(time_scale=100.0)
        self.env = default_registry[cfg.env].make(seed=self.seed, worker_id=cfg.work_id, side_channels=[channel], no_graphics=True)
        self.env.reset()
        self.behavior_name = list(self.env.behavior_specs)[0]

        eval_channel = EngineConfigurationChannel()
        eval_channel.set_configuration_parameters(time_scale=100.0)
        self.eval_env = default_registry[cfg.env].make(seed=self.seed+1, worker_id=cfg.work_id_ev, side_channels=[eval_channel], no_graphics=True)
        self.eval_env.reset()
        self.eval_behavior_name = list(self.eval_env.behavior_specs)[0]

        self.spec = self.env.behavior_specs[self.behavior_name]
        cfg.model.params.obs_shape = sum(i.shape[0] for i in self.spec.observation_specs)
        if self.spec.action_spec.is_continuous():
            cfg.model.params.num_actions = self.spec.action_spec.continuous_size
        elif self.spec.action_spec.is_discrete():
            cfg.model.params.num_actions = sum(list(self.spec.action_spec.discrete_branches))
            cfg.model.params.num_actions_branch = self.spec.action_spec.discrete_size

        self.num_agents = cfg.num_agents

        self.model = hydra.utils.instantiate(cfg.model)
        self.model.logger = self.logger
        self.model.on_begin()

        self.step = 0
        self.batch_size = cfg.batch_size
        self.agent_eps = 0.0

        self.average_episode_reward = 0

        wandb.init(project="Soccer2", entity="zhengtr")

    def evaluate(self):
        self.average_episode_reward = 0
        eval_step = 0
        num_eval_episodes = 0
        while eval_step < self.cfg.num_eval_steps:
            # print(f'eval_step: {eval_step}')
            self.eval_env.reset()
            decision_steps, terminal_steps = self.eval_env.get_steps(self.eval_behavior_name)
            num_agents = len(decision_steps)

            episode_reward = 0
            episode_step = 0
            while decision_steps or not terminal_steps:  # no goal yet
                # print(f'eval_episode_step: {episode_step}')
                if np.random.rand() < self.model.eval_eps:
                    action = self.spec.action_spec.random_action(len(decision_steps))
                else:
                    obs = torch.FloatTensor(np.array([np.concatenate(decision_steps[i].obs) for i in decision_steps.agent_id]))
                    action_np = np.array(self.model.get_actions(obs, self.eps, eval_flag=True))
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
            # self.video_recorder.save(f'{num_eval_episodes}.mp4')
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
                    wandb.log(
                        {"Train Episodic Return": episode_reward, "Average Episodic Return": self.average_episode_reward}, step=self.step)

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
            self.eps = self.cfg.min_eps + bonus

            self.logger.log('train/eps', self.eps, self.step)
            obs = torch.FloatTensor(np.array([np.concatenate(decision_steps[i].obs) for i in decision_steps.agent_id]))
  
            action_np = np.array(self.model.get_actions(obs, self.eps, eval_flag=True))
            action = ActionTuple()
            action.add_discrete(action_np)

 
            self.env.set_actions(self.behavior_name, action)
            self.env.step()
            next_decision_steps, next_terminal_steps = self.env.get_steps(self.behavior_name)
            
            obs_batch = []  # [ [states of agent 1], ... ,[states of agent n] ]    
            indiv_action_batch = [] # [ [actions of agent 1], ... , [actions of agent n]]
            indiv_reward_batch = []
            next_obs_batch = []
            done_batch = []
            
            for agent_i in next_terminal_steps.agent_id:
                reward = next_terminal_steps[agent_i].reward
                pre_obs = decision_steps[agent_i].obs
                index = decision_steps.agent_id_to_index[agent_i]
                act = torch.FloatTensor(action.discrete[index])

                obs_batch.append(np.concatenate(pre_obs))
                indiv_action_batch.append(act)
                indiv_reward_batch.append(reward)
                next_obs_batch.append(np.concatenate(next_terminal_steps[agent_i].obs))
                done_batch.append(float(True))

                episode_reward += reward

            
            for agent_i in next_decision_steps.agent_id:
                if len(obs_batch) < self.num_agents:
                    reward = next_decision_steps[agent_i].reward
                    index = next_decision_steps.agent_id_to_index[agent_i]
                    act = torch.FloatTensor(action.discrete[index])

                    obs_batch.append(np.concatenate(decision_steps[agent_i].obs))
                    indiv_action_batch.append(act)
                    indiv_reward_batch.append(reward)
                    next_obs_batch.append(np.concatenate(next_decision_steps[agent_i].obs))
                    done_batch.append(float(False))

                    episode_reward += reward


            if len(obs_batch) == 16:
                self.model.replay_buffer.push(np.array(obs_batch), indiv_action_batch, indiv_reward_batch, np.array(next_obs_batch), done_batch)
            # else:
            #     print(f'len_obs_batch: {len(obs_batch)}')

            if len(self.model.replay_buffer) > self.batch_size:
                self.model.train()

            decision_steps = next_decision_steps
            terminal_steps = next_terminal_steps

            episode_step += 1
            self.step += 1
            self.model.step = self.step
            # print(f'step: {self.step}')


        self.env.close()
        self.eval_env.close()
        self.model.on_exit()

@hydra.main(config_path='config.yaml')
def main(cfg):
    from train import Workspace as W
    workspace = W(cfg)
    workspace.run()


if __name__ == '__main__':
    main()