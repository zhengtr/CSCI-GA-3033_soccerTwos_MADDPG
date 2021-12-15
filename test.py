from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.registry import default_registry
import torch
import numpy as np
from mlagents_envs.side_channel.engine_configuration_channel import EngineConfigurationChannel
from mlagents_envs.environment import ActionTuple

try:
  env.close()
except:
  pass
channel = EngineConfigurationChannel()


env = default_registry['SoccerTwos'].make(side_channels=[channel])
channel.set_configuration_parameters(width=2000, height=1000)
# env.close()
# env = UnityEnvironment(file_name="Soccer", seed=1, side_channels=[])
env.reset()
behavior_name = list(env.behavior_specs)[0]
# behavior_name2 = list(env.behavior_specs)[1]
k = env.behavior_specs
print(f"Name of the behavior : {behavior_name}")
spec = env.behavior_specs[behavior_name]
o = sum(i.shape[0] for i in spec.observation_specs)
# [0].shape
# print("Number of observations : ", *o)
# a = torch.zeros(1, *o[0].shape)
# print("Number of observations : ", a.size())
vis_obs = any(len(spec.shape) == 3 for spec in spec.observation_specs)
print("Is there a visual observation ?", vis_obs)


if spec.action_spec.continuous_size > 0:
  print(f"There are {spec.action_spec.continuous_size} continuous actions")
if spec.action_spec.is_discrete():
  print(f"There are {spec.action_spec.discrete_size} discrete actions")

if spec.action_spec.discrete_size > 0:
  for action, branch_size in enumerate(spec.action_spec.discrete_branches):
    print(f"Action number {action} has {branch_size} different options")

# decision_steps, terminal_steps = env.get_steps(behavior_name)
# env.set_actions(behavior_name, spec.action_spec.empty_action(len(decision_steps)))
# env.step()

for episode in range(3):
  env.reset()
  decision_steps, terminal_steps = env.get_steps(behavior_name)
  tracked_agent = -1 # -1 indicates not yet tracking
  done = False # For the tracked_agent
  episode_rewards = 0 # For the tracked_agent
  while not done:
    # Track the first agent we see if not tracking
    # Note : len(decision_steps) = [number of agents that requested a decision]
    if tracked_agent == -1 and len(decision_steps) >= 1:
      tracked_agent = decision_steps.agent_id[1]

    # Generate an action for all agents
    action = spec.action_spec.random_action(len(decision_steps))
    action.discrete[2:] = [[0,0,0]] * 14
    # action.discrete[4:] = [[0, 0, 0]] * 12
    # actionN = ActionTuple()
    # actionN.add_discrete(action)
    # Set the actions
    env.set_actions(behavior_name, action)

    # Move the simulation forward
    k = env.step()

    # Get the new simulation results
    decision_steps, terminal_steps = env.get_steps(behavior_name)
    if tracked_agent in decision_steps: # The agent requested a decision
      episode_rewards += decision_steps[tracked_agent].reward
    if tracked_agent in terminal_steps: # The agent terminated its episode
      episode_rewards += terminal_steps[tracked_agent].reward
      done = True
  print(f"Tot al rewards for episode {episode} is {episode_rewards}")

