import torch
import pandas as pd
import numpy as np
import replay
import soft_actor_critic



# Load networks
qf1         = torch.load('flight5-qf1.pt')
qf2         = torch.load('flight5-qf2.pt')
qf1_target  = torch.load('flight5-qf1_target.pt')
qf2_target  = torch.load('flight5-qf2_target.pt')
policy      = torch.load('flight5-policy.pt')

networks = {'qf1' : qf1, 'qf2' : qf2, 'qf1_target' : qf1_target, 'qf2_target' : qf2_target, 'policy' : policy}

# Flight data to replay buffer
folder_path = './SAC-goalbased/policy-flight-data/'
# file_path = str(sys.argv[1])
date = 'Wed-Aug--2-16:30:24-2023'
state_path   = folder_path + 'state_list-' + date + '.csv'
actions_path = folder_path + 'action_list-' + date + '.csv'

state_data = pd.read_csv(state_path)
actions_data = pd.read_csv(actions_path)

state_data_np = state_data.to_numpy()
goal_positions = state_data_np[:, 19:24] 
stable_orientations = state_data_np[:, 24:28]
states = state_data_np[:, 1:]
actions = actions_data.to_numpy()
actions = actions[:, 1:]

# Normalise actions
action_means_stds = pd.read_csv('./SAC-goalbased/jul31-action_means_stds.csv')

def clip_and_norm_actions(actions):
    # Clip and normalise actions through Z-score
    means = action_means_stds['Action means']
    stds  = action_means_stds['Action stds']

    for i in range(np.shape(actions)[1]):
        actions[:,i] = (actions[:,i] - means[i]) / stds[i]
        actions[:,i] = np.clip(actions[:,i], -1, 1)

    return actions

actions = clip_and_norm_actions(actions)
print('normalised actions', actions)

# print('goal pos', goal_positions)
# print('goal or', stable_orientations)
# print('states', states)

observation_dim = np.shape(states)[1]
action_dim = np.shape(actions)[1]

def calculate_rewards(state, goal_positions, stable_orientation):
    # Calculate rewards using error between current state and goal state
    # Goal state defined by an arbitrary height, 0 velocity and orientation from when the drone is on the ground
    # --> maximise -sqrt( (curr pos - goal pos)**2 + (curr orientation - stable orientation)**2 )
    # wrap it in an exponential to ensure rewards stay small

    # TODO: only include pitch & roll in orientation error?
    # TODO: should different parts of reward be weighted differently?
    # TODO: penalty for large actions?

    pos_error = np.array([state[:,0], state[:,1], state[:,2], state[:,3], state[:,4]]).transpose() - goal_positions
    orientation_error = np.array([state[:,5], state[:,6], state[:,7], state[:,8]]).transpose() - stable_orientation

    print(state[100], goal_positions[100])
    print('pos error', pos_error)
    print('or error', orientation_error)

    # Take the norm of each error vector separately to get a vector of rewards https://stackoverflow.com/questions/7741878/how-to-apply-numpy-linalg-norm-to-each-row-of-a-matrix
    rewards = np.exp( - (np.sum(np.abs(pos_error)**2, axis = -1)**(1./2) + np.sum(np.abs(orientation_error)**2, axis = -1)**(1./2)))
    rewards *= 100

    return rewards

rewards = calculate_rewards(states, goal_positions, stable_orientations)
print('rewards', rewards)

# save rewards to file
rewards_df = pd.DataFrame(rewards)
rewards_df.to_csv(folder_path + 'rewards-' + date + '.csv')

# print(np.shape(states))
# print(np.shape)

# initialise replay buffer
replay = replay.SimpleReplayBuffer(
            max_replay_buffer_size=1000000,
            observation_dim=observation_dim,
            action_dim=action_dim,
            env_info_sizes={},)

# print(np.shape(states))
# print(np.shape(actions))
# print(np.shape(rewards))

for i in range(np.shape(states)[0] - 1):
    replay.add_sample(observation=states[i], 
                      action=actions[i], 
                      reward=rewards[i], 
                      next_observation=states[i+1], 
                      terminal=0, 
                      env_info={})
    # print(states[i])
    # print(actions[i])
    # print(rewards[i])
    # print(states[i+1])



agent = soft_actor_critic.SoftActorCritic(replay=replay, networks=networks)

print(agent._networks['policy']) # 27 inputs, 27 states
print(agent._networks['qf1']) # 31 inputs, 27 states + 4 actions
print(agent._networks['qf2']) # 31 inputs, 27 states + 4 actions
print(agent._networks['qf1_target']) # 31 inputs, 27 states + 4 actions
print(agent._networks['qf2_target']) # 31 inputs, 27 states + 4 actions

print(observation_dim)
print(action_dim)

for i in range(100):
    agent.single_train_step()
    
    statistics = agent._algorithm.get_diagnostics()
    print('Training step: ', i)
    print(statistics)


# Save policy and Q-function networks torch objects for future use
# TODO: naming
torch.save(agent._algorithm.policy,     'flight6-policy.pt')
torch.save(agent._algorithm.qf1,        'flight6-qf1.pt')
torch.save(agent._algorithm.qf2,        'flight6-qf2.pt')
torch.save(agent._algorithm.target_qf1, 'flight6-qf1_target.pt')
torch.save(agent._algorithm.target_qf2, 'flight6-qf2_target.pt')



