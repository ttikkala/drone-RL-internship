import replay
import os
import time
import numpy as np
import pandas as pd
import soft_actor_critic
import csv
import torch
import replay
import time
import matplotlib.pyplot as plt
import drawnow

"""
Main file for training the SAC algorithm.

First imports data, then initialises the replay buffer and the networks.
Then the algortihm is trained using rlkit's implementation of SAC.
The training stats are written to file. 

"""




# Import data
# TODO: Change this to be a command line argument, wrap in function or something? Kind of messy like this
folder_path = './'
# file_path = str(sys.argv[1])
state_path   = folder_path + 'states-no-vel.csv'
rewards_path = folder_path + 'rewards-no-vel.csv'
actions_path = folder_path + 'actions-no-vel.csv'

state_data = pd.read_csv(state_path)
rewards_data = pd.read_csv(rewards_path)
actions_data = pd.read_csv(actions_path)

states = state_data.to_numpy()
rewards = rewards_data.to_numpy()
actions = actions_data.to_numpy()


observation_dim = np.shape(states)[1]
action_dim = np.shape(actions)[1]

# initialise replay buffer
replay = replay.SimpleReplayBuffer(
            max_replay_buffer_size=1000000,
            observation_dim=observation_dim,
            action_dim=action_dim,
            env_info_sizes={},)

# initialise networks qf1, qf2, qf1_target, qf2_target, policy
networks = soft_actor_critic.SoftActorCritic._create_networks(obs_dim=observation_dim, action_dim=action_dim)

# Initialise SAC agent
agent = soft_actor_critic.SoftActorCritic(replay=replay, networks=networks)



# Populate replay buffer with collected flight data
# TODO: Figure out if terminal state matters for real-world data
for i in range(np.shape(states)[0] - 1):
    replay.add_sample(observation=states[i], 
                      action=actions[i], 
                      reward=rewards[i], 
                      next_observation=states[i+1], 
                      terminal=0, 
                      env_info={})


# File path for training results
file_path = 'losses-' + time.ctime().replace(' ', '-') + '.csv'
with open(file_path, 'a') as fd:
    cwriter = csv.writer(fd)
    cwriter.writerow(['Training step', 'QF1 Loss', 'QF2 Loss', 
                      'Policy Loss', 'Q1 Predictions Mean', 'Q1 Predictions Std', 'Q1 Predictions Max', 
                      'Q2 Predictions Min', 'Q Targets Mean', 'Q Targets Std', 'Q Targets Max', 'Q Targets Min', 
                      'Log Pis Mean', 'Log Pis Std', 'Log Pis Max', 'Log Pis Min', 
                      'Policy mu Mean', 'Policy mu Std', 'Policy mu Max', 'Policy mu Min', 
                      'Policy log std Mean', 'Policy log std Std', 'Policy log std Max', 'Policy log std Min']) 




try:
    # Start training
    for i in range(1000):
        agent.single_train_step()

        # Print training stats; losses etc
        statistics = agent._algorithm.get_diagnostics()
        print('Training step: ', i)
        print(statistics)


        # Training stats to file
        with open(file_path, 'a') as fd:
            cwriter = csv.writer(fd)
            cwriter.writerow([i, statistics['QF1 Loss'], statistics['QF2 Loss'], 
                        statistics['Policy Loss'], statistics['Q1 Predictions Mean'], statistics['Q1 Predictions Std'], statistics['Q1 Predictions Max'], 
                        statistics['Q2 Predictions Min'], statistics['Q Targets Mean'], statistics['Q Targets Std'], statistics['Q Targets Max'], statistics['Q Targets Min'], 
                        statistics['Log Pis Mean'], statistics['Log Pis Std'], statistics['Log Pis Max'], statistics['Log Pis Min'], 
                        statistics['Policy mu Mean'], statistics['Policy mu Std'], statistics['Policy mu Max'], statistics['Policy mu Min'], 
                        statistics['Policy log std Mean'], statistics['Policy log std Std'], statistics['Policy log std Max'], statistics['Policy log std Min']])
            

except KeyboardInterrupt:
    # Save policy and Q-function networks torch objects for future use in case of ctrl+C
    torch.save(agent._algorithm.policy,     'aug07-novel-policy.pt')
    torch.save(agent._algorithm.qf1,        'aug07-novel-qf1.pt')
    torch.save(agent._algorithm.qf2,        'aug07-novel-qf2.pt')
    torch.save(agent._algorithm.target_qf1, 'aug07-novel-qf1_target.pt')
    torch.save(agent._algorithm.target_qf2, 'aug07-novel-qf2_target.pt')



# Save policy and Q-function networks torch objects for future use
torch.save(agent._algorithm.policy,     'aug07-novel-policy.pt')
torch.save(agent._algorithm.qf1,        'aug07-novel-qf1.pt')
torch.save(agent._algorithm.qf2,        'aug07-novel-qf2.pt')
torch.save(agent._algorithm.target_qf1, 'aug07-novel-qf1_target.pt')
torch.save(agent._algorithm.target_qf2, 'aug07-novel-qf2_target.pt')
