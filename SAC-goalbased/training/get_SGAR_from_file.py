import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import csv
import sys
import os

"""
This file calculates states, goals, actions and rewards from lab data.

First syncs data between drone and mocap, then combines them into states and actions, which are
normalised and saved to csv files. The state is augmented with the goal position.
Then calculates rewards based on a goal position and orientation,
and saves to csv file.
"""



# TODO: Change these to command line arguments, maybe merge all flight data into one file?
# Get flight input and mocap data
folder_path_drone = '20230706T16-03-59/'
file_path_battery = './SAC-goalbased/training/' + folder_path_drone + 'Battery-20230706T16-04-09-timestamped.csv'
file_path_motors  = './SAC-goalbased/training/' + folder_path_drone + 'Motors_fast-20230706T16-04-08-timestamped.csv'
file_path_stab    = './SAC-goalbased/training/' + folder_path_drone + 'Stab-20230706T16-04-07-timestamped.csv'

folder_path_mocap = './SAC-goalbased/training/mocap_data/'
file_path_mocap = folder_path_mocap + 'Thu_Jul__6_16:04:18_2023/data.csv'

battery_data = pd.read_csv(file_path_battery)
motors_data = pd.read_csv(file_path_motors)
stab_data = pd.read_csv(file_path_stab)

mocap_data = pd.read_csv(file_path_mocap)

# Use mocap data as base for syncing since it has highest data rate
# Sync data sets by linear interpolation
action_pitch    = - np.interp(mocap_data['Timestamp'], stab_data['Timestamp'], stab_data['stabilizer.pitch'])
action_roll     = np.interp(mocap_data['Timestamp'], stab_data['Timestamp'], stab_data['stabilizer.roll'])
action_yaw      = np.interp(mocap_data['Timestamp'], stab_data['Timestamp'], stab_data['stabilizer.yaw'])
action_thrust   = np.interp(mocap_data['Timestamp'], stab_data['Timestamp'], stab_data['stabilizer.thrust'])
state_battery   = np.interp(mocap_data['Timestamp'], battery_data['Timestamp'], battery_data['pm.vbat'])
state_motor1    = np.interp(mocap_data['Timestamp'], motors_data['Timestamp'], motors_data['motor.m1'])
state_motor2    = np.interp(mocap_data['Timestamp'], motors_data['Timestamp'], motors_data['motor.m2'])
state_motor3    = np.interp(mocap_data['Timestamp'], motors_data['Timestamp'], motors_data['motor.m3'])
state_motor4    = np.interp(mocap_data['Timestamp'], motors_data['Timestamp'], motors_data['motor.m4'])

# Get rest of state data from mocap
state_posx  = mocap_data['Pos x'].to_numpy()
state_posy  = mocap_data['Pos y'].to_numpy()
state_posz  = mocap_data['Pos z'].to_numpy()
state_quatx = mocap_data['Quat w'].to_numpy() # Incorrectly labelled in mocap data
state_quaty = mocap_data['Quat x'].to_numpy()
state_quatz = mocap_data['Quat y'].to_numpy()
state_quatw = mocap_data['Quat z'].to_numpy()

# Get x and z relative to middle of mat
# TODO: Not sure if this is correct
state_posx = state_posx - 3.675
state_posy = state_posy
state_posz = state_posz - 2.712

# Take derivatives to get velocities and angular velocities
state_vx        = np.gradient(state_posx, mocap_data['Timestamp'])
state_vy        = np.gradient(state_posy, mocap_data['Timestamp'])
state_vz        = np.gradient(state_posz, mocap_data['Timestamp'])

print('vx', state_vx, np.shape(state_vx))
print('vz', state_vz, np.shape(state_vz))
vel_effects = True
if vel_effects:
    with np.printoptions(threshold=50000):
        print(state_vx)
    state_vx = np.where(np.abs(state_vx) < 0.05, 0.0, state_vx)
    # state[:,4] = np.where(np.abs(state[:,4]) < 0.05, 0.0, state[:,4])
    with np.printoptions(threshold=50000):
        print(state_vx)
    

state_angvelx   = np.gradient(state_quatx, mocap_data['Timestamp'])
state_angvely   = np.gradient(state_quaty, mocap_data['Timestamp'])
state_angvelz   = np.gradient(state_quatz, mocap_data['Timestamp'])
state_angvelw   = np.gradient(state_quatw, mocap_data['Timestamp'])
action_yawrate  = np.gradient(action_yaw, mocap_data['Timestamp'])

# Plotting interpolated values to sanity check
plotting = False
if plotting:
    plt.subplot(2, 1, 1)
    plt.plot(motors_data['Timestamp'], motors_data['motor.m1'], label='pitch')
    plt.title('')
    plt.ylabel('M1')
    plt.xlabel('Time')

    plt.subplot(2, 1, 2)
    plt.plot(mocap_data['Timestamp'], state_motor1, label='interp pitch')
    plt.ylabel('Interpolated M1')
    plt.xlabel('Time')

    plt.show()


def clip_and_norm_actions(actions):
    # Clip and normalise actions through Z-score
    means = []
    stds = []

    for i in range(np.shape(actions)[1]):
        mean = np.mean(actions[:,i])
        std = np.std(actions[:,i])
        means.append(mean)
        stds.append(std)
        actions[:,i] = (actions[:,i] - mean) / std
        actions[:,i] = np.clip(actions[:,i], -1, 1)

    return actions, means, stds

def normalise_state(state):
    # Normalise states through Z-score
    means = []
    stds = []

    for i in range(np.shape(state)[1]):
        mean = np.mean(state[:,i])
        std = np.std(state[:,i])
        means.append(mean)
        stds.append(std)
        state[:,i] = (state[:,i] - mean) / std 
        state[:,i] = np.clip(state[:,i], -1, 1)
        state[:,i] *= 5

    return state, means, stds

def normalise_goal_pos(goal_pos, means, stds):
    # Normalise states through Z-score
    # goal_pos is a 1D array [pos_y, vel_x, vel_z]
    temp = np.zeros(5)

    for i in range(5):
        # print(goal_pos[i], means[i], stds[i])
        temp[i] = (goal_pos[i] - means[i]) / stds[i]
        # print(goal_pos[i])
        temp[i] = np.clip(goal_pos[i], -1, 1)
        temp[i] *= 5

    return temp

def calculate_rewards(state, goal_positions, stable_orientation):
    # Calculate rewards using error between current state and goal state
    # Goal state defined by an arbitrary height, 0 velocity and orientation from when the drone is on the ground
    # --> maximise -sqrt( (curr pos - goal pos)**2 + (curr orientation - stable orientation)**2 )
    # wrap it in an exponential to ensure rewards stay small

    # TODO: only include pitch & roll in orientation error?
    # TODO: should different parts of reward be weighted differently?
    # TODO: penalty for large actions?

    # TODO: try without velocity term, or with a penalty for large velocities

    pos_error = np.array([state[:,0], state[:,1], state[:,2], np.zeros(np.shape(state[:,3])[0]), np.zeros(np.shape(state[:,3])[0])]).transpose() - goal_positions
    orientation_error = np.array([state[:,5], state[:,6], state[:,7], state[:,8]]).transpose() - stable_orientation

    # Take the norm of each error vector separately to get a vector of rewards https://stackoverflow.com/questions/7741878/how-to-apply-numpy-linalg-norm-to-each-row-of-a-matrix
    rewards = np.exp( - (np.sum(np.abs(pos_error)**2, axis = -1)**(1./2) + np.sum(np.abs(orientation_error)**2, axis = -1)**(1./2)))
    # rewards *= 1.0/100

    return rewards


def choose_goal_position(means, stds, length):

    hover_range = np.arange(0.1, 0.4, 0.05)
    plane_pos_range = np.arange(-0.5, 0.5, 0.05)

    goal_position_array = []
    goal_position = np.array([np.random.choice(plane_pos_range), np.random.choice(hover_range), np.random.choice(plane_pos_range), 0.0, 0.0]) # [rel x, y, rel z, vel_x, vel_z]
    # print(goal_position)

    for i in range(length):            
        if i % 500 == 0:
            goal_position = np.array([np.random.choice(plane_pos_range), np.random.choice(hover_range), np.random.choice(plane_pos_range), 0.0, 0.0])
            # print(goal_position)
        goal_position_array.append(normalise_goal_pos(goal_position, means, stds))

    return np.array(goal_position_array)

# def choose_goal_position(length):

#     hover_range = np.arange(0.1, 0.4, 0.05)
#     plane_pos_range = np.arange(-0.5, 0.5, 0.05)

#     goal_position_array = []
#     goal_position = np.array([np.random.choice(plane_pos_range), np.random.choice(hover_range), np.random.choice(plane_pos_range), 0.0, 0.0]) # [rel x, y, rel z, vel_x, vel_z]
#     # print(goal_position)

#     for i in range(length):
#         if i % 500 == 0:
#             goal_position = np.array([np.random.choice(plane_pos_range), np.random.choice(hover_range), np.random.choice(plane_pos_range), 0.0, 0.0])
#             # print(goal_position)
#         goal_position_array.append(goal_position)

#     return np.array(goal_position_array)


def main():

    # Get states, actions and rewards and save to csv for training

    # Actions: commander inputs to CF firmware as (roll, pitch, yawrate, thrust)
    action_data = np.column_stack((action_roll, action_pitch, action_yawrate, action_thrust))
    
    action_data, ac_means, ac_stds = clip_and_norm_actions(action_data)

    output_file = "./" + "actions-large-vel.csv"
    action_df = pd.DataFrame({'roll commands': action_data[:,0], 
                              'pitch commands': action_data[:,1], 
                              'yawrate commands': action_data[:,2], 
                              'thrust commands': action_data[:,3]})
    action_df.to_csv(output_file, index=False)


    # State: relative position (rel x, y, rel z, v_x, v_z), orientation (quaternions x, y, z, w), angular velocities, current motor PWM values, battery voltage
    state_data = np.column_stack((state_posx, state_posy, state_posz, 
                                state_vx, state_vz, 
                                state_quatx, state_quaty, state_quatz, state_quatw, 
                                state_angvelw, state_angvelx, state_angvely, state_angvelz, 
                                state_motor1, state_motor2, state_motor3, state_motor4, 
                                state_battery))
    

    # Normalise states
    state_data, state_means, state_stds = normalise_state(state_data)
    print('norm vx', state_data[:,3])
    print('norm vz', state_data[:,4])

    # TODO: Make goal position and orientation be calculated by a function
    # Define goal position and orientation
    # Normalise goal altitude, 0 velocity in x and z 
    # Goal altitude is decided randomly between 0.3 and 0.6 m, velocity in x and z stays 0
    # goal_position = np.array([((goal_alt - state_means[0]) / state_stds[0]), ((0.0 - state_means[1]) / state_stds[1]), ((0.0 - state_means[2]) / state_stds[2])]) # [pos_y, vel_x, vel_z]
    goal_positions = choose_goal_position(state_means, state_stds, length=np.shape(state_data)[0])
    # goal_positions = choose_goal_position(length=np.shape(state_data)[0])
    # print(goal_positions)
    


    # Goal orientation when the drone is on a flat surface, average of first 10 samples, normalised
    # TODO: I think this can just stay how it is? I don't think it's particularly interesting to change the goal orientation?
    # stable_orientation = np.array([(np.mean(state_data[:10,7]) - state_means[3]) / state_stds[3], 
    #                                (np.mean(state_data[:10,8]) - state_means[4]) / state_stds[4], 
    #                                (np.mean(state_data[:10,9]) - state_means[5]) / state_stds[5], 
    #                                (np.mean(state_data[:10,10]) - state_means[6]) / state_stds[6]])
    stable_orientation = np.array([np.mean(state_data[:10,7]), 
                                   np.mean(state_data[:10,8]), 
                                   np.mean(state_data[:10,9]), 
                                   np.mean(state_data[:10,10])])
    
    # print('stable ', stable_orientation)

    # print(np.shape(state_data[0]))
    # print(np.shape(goal_positions))
    # print(goal_positions)

    # In this version the state is augmented with the goal position and orientation
    output_file = "./" + "states-large-vel.csv"
    state_df = pd.DataFrame({'Rel pos x': state_data[:,0], 'Rel pos y': state_data[:,1], 'Rel pos z': state_data[:,2], 
                            'Vel x': state_data[:,3], 'Vel z': state_data[:,4], 
                            'Quat x': state_data[:,5], 'Quat y': state_data[:,6], 'Quat z': state_data[:,7], 'Quat w': state_data[:,8],
                            'Ang vel x': state_data[:,9], 'Ang vel y': state_data[:,10], 'Ang vel z': state_data[:,11], 'Ang vel w': state_data[:,12],
                            'motor.m1' : state_data[:,13], 'motor.m2' : state_data[:,14], 'motor.m3' : state_data[:,15], 'motor.m4' : state_data[:,16],
                            'battery' : state_data[:,17],
                            'goal pos x': goal_positions[:,0], 'goal pos y': goal_positions[:,1], 'goal pos z': goal_positions[:,2],
                            'goal vel x': goal_positions[:,3], 'goal vel z': goal_positions[:,4],
                            'goal quat x': np.ones(np.shape(state_data)[0]) * stable_orientation[0], 
                            'goal quat y': np.ones(np.shape(state_data)[0]) * stable_orientation[1], 
                            'goal quat z': np.ones(np.shape(state_data)[0]) * stable_orientation[2], 
                            'goal quat w': np.ones(np.shape(state_data)[0]) * stable_orientation[3]})
    state_df.to_csv(output_file, index=False)




    # Calculate rewards for each state
    rewards = calculate_rewards(state_data, goal_positions, stable_orientation)

    output_file = "./" + "rewards-large-vel.csv"
    reward_df = pd.DataFrame({'Rewards': rewards})
    reward_df.to_csv(output_file, index=False)


    # Save actions means, stds and state means, stds to csv
    output_file = "./" + "action_means_stds-large-vel.csv"
    ac_means_stds_df = pd.DataFrame({'Action means': ac_means, 'Action stds': ac_stds})
    ac_means_stds_df.to_csv(output_file, index=False)
    output_file = "./" + "state_means_stds-large-vel.csv"
    state_means_stds_df = pd.DataFrame({'State means': state_means, 'State stds': state_stds})
    state_means_stds_df.to_csv(output_file, index=False)





if __name__=='__main__':
    main()


