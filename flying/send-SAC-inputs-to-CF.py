import time
import pandas as pd
import numpy as np
import csv
import os
import torch
import natnet
import attr
import argparse

import cflib.crtp
from cflib.crazyflie import Crazyflie
from cflib.crazyflie.syncCrazyflie import SyncCrazyflie
from cflib.utils import uri_helper
from cflib.crazyflie.log import LogConfig
from cflib.crazyflie.syncLogger import SyncLogger
import logging

import threading


###
# This file contains the code for sending the SAC inputs to the Crazyflie
#
# The rough outline of the code is as follows:
# Initialise threads; one for drone data and control, one for OptiTrack data, one for policy network
# Get state in real-time from OptiTrack and drone
# Transform state to normalised state and clip
# Get action from policy
# Transform action to drone command
# Send command to drone
# TODO: Update policy based on reward after each flight
###




# Load policy that was trained using SAC/main.py
policy = torch.load('./SAC/training/jul14-policy.pt')

# Initialise values used for SAC calculations
t_prev = time.time()
x_prev = 0.0
z_prev = 0.0
qx_prev = 0.0
qy_prev = 0.0
qz_prev = 0.0
qw_prev = 0.0

# Get means and stds used for SAC normalisation from file
state_means_stds = pd.read_csv('./SAC/training/jul14-state_means_stds.csv')
action_means_stds = pd.read_csv('./SAC/training/jul14-action_means_stds.csv')

###### CRAZYFLIE DRONE CODE ######
# Crazyflie initialise radio connection
# uri = uri_helper.uri_from_env(default='radio://0/80/2M/E7E7E7E7E7')
uri = uri_helper.uri_from_env(default='radio://0/100/2M/E7E7E7E7E7')

# Only output errors from the logging framework
logging.basicConfig(level=logging.ERROR)

drone_signals = [0, 0, 0, 0, 0]

def log_callback(timestamp, data, logconf):
    # print(data)
    global drone_signals
    global drone_reader
    
    drone_signals[0] = data['motor.m1']
    drone_signals[1] = data['motor.m2']
    drone_signals[2] = data['motor.m3']
    drone_signals[3] = data['motor.m4']
    drone_signals[4] = data['pm.vbat']

    # print('Drone signals in: ', [drone_signals[0], drone_signals[1], drone_signals[2], drone_signals[3], drone_signals[4]])

    drone_reader.read_data(drone_signals)


def command_from_network(scf):
    global sac_reader

    time.sleep(4.0)

    while True:
        sac_reader.lock.acquire()
        try:
            action = sac_reader.value
            # print('Drone lock acquired, drone data: ', drone_data)
        finally:
            sac_reader.lock.release()
            time.sleep(0.02) # 50 Hz

        print('Command: ', action[0], action[1], action[2], int(action[3]))
        # TODO: -pitch or pitch?
        # time.sleep(0.1)
        scf.cf.commander.send_notify_setpoint_stop(10)
        scf.cf.commander.send_setpoint(action[0], action[1], action[2], int(action[3])) # roll, pitch, yawrate, thrust



# CF flight code
def fly_drone():

    # Initialize the low-level drivers
    cflib.crtp.init_drivers()

    # TODO: Check if this many variables fit in the data stream
    log = LogConfig(name='Data', period_in_ms=10)
    # log.add_variable('stabilizer.thrust', 'float')
    # log.add_variable('stabilizer.roll', 'float')
    # log.add_variable('stabilizer.pitch', 'float')
    # log.add_variable('stabilizer.yaw', 'float')
    log.add_variable('motor.m1', 'float')
    log.add_variable('motor.m2', 'float')
    log.add_variable('motor.m3', 'float')
    log.add_variable('motor.m4', 'float')
    log.add_variable('pm.vbat', 'float')


    with SyncCrazyflie(uri, cf=Crazyflie(rw_cache='./cache')) as scf:

        scf.cf.log.add_config(log)
        log.data_received_cb.add_callback(log_callback)

        log.start()

        scf.cf.commander.send_setpoint(0, 0, 0, 0)

        time.sleep(1.5)

        command_from_network(scf)

        scf.cf.commander.send_setpoint(0, 0, 0, 0)

        log.stop()

        time.sleep(0.1)
        scf.cf.close_link()
    

###### OPTITRACK CODE ######
# Natnet SDK connection to optitrack data stream
@attr.s
class ClientApp(object):

    _client = attr.ib()
    _quiet = attr.ib()

    _last_printed = attr.ib(0)

    @classmethod
    def connect(cls, server_name, rate, quiet):
        if server_name == 'fake':
            client = natnet.fakes.SingleFrameFakeClient.fake_connect(rate=rate)
        else:
            client = natnet.Client.connect(server_name)
        if client is None:
            return None
        return cls(client, quiet)

    def run(self):
        if self._quiet:
            self._client.set_callback(self.callback_quiet)
        else:
            self._client.set_callback(self.callback)
        self._client.spin()

    def callback(self, rigid_bodies, markers, timing):
        """
        :type rigid_bodies: list[RigidBody]
        :type markers: list[LabelledMarker]
        :type timing: TimestampAndLatency
        """
        # print()
        # print('{:.1f}s: Received mocap frame'.format(timing.timestamp))
        global opti_reader

        if rigid_bodies:
            # print('Rigid bodies:')
            for b in rigid_bodies:
            #     print('\t Id {}: ({: 5.2f}, {: 5.2f}, {: 5.2f}), ({: 5.2f}, {: 5.2f}, {: 5.2f}, {: 5.2f})'.format(
            #         b.id_, *(b.position + b.orientation)
            #     ))

                opti_reader.read_data([b.id_, *(b.position + b.orientation)])

    def callback_quiet(self, *_):
        if time.time() - self._last_printed > 1:
            print('.')
            self._last_printed = time.time()


def natnet_main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--server', help='Will autodiscover if not supplied')
    parser.add_argument('--fake', action='store_true',
                        help='Produce fake data at `rate` instead of connecting to actual server')
    parser.add_argument('--rate', type=float, default=10,
                        help='Rate at which to produce fake data (Hz)')
    parser.add_argument('--quiet', action='store_true')
    args = parser.parse_args()

    try:
        app = ClientApp.connect('fake' if args.fake else args.server, args.rate, args.quiet)
        app.run()
    except natnet.DiscoveryError as e:
        print('Error:', e)

###### THREADING DATA SHARING OBJECT ######
class DataReader(object):

    def __init__(self, start = []):
        self.lock = threading.Lock()
        self.value = start

    def read_data(self, data_in):
        logging.debug('Waiting for a lock')
        self.lock.acquire()
        try:
            logging.debug('Acquired a lock')
            self.value = data_in
            # print('Data in: ', self.value)
        finally:
            logging.debug('Released a lock')
            self.lock.release()
        
        # time.sleep(0.01)

###### SAC CODE ######
def normalise_state(state):
    global state_means_stds

    for i in range(np.shape(state)[0]):
        state[i] = (state[i] - state_means_stds['State means'][i]) / state_means_stds['State stds'][i] 
        state[i] = np.clip(state[i], -1, 1)
        state[i] *= 5

    return state


def action_to_drone_command(action):
    global action_means_stds

    # Action is input as a torch tensor, detach to get numpy array
    action = action[0].detach().cpu().numpy()

    # Transform action from normalised value to a real drone command
    # x = (Z * std) + mean
    for i in range(np.shape(action)[0]):
        action[i] = (action[i] * action_means_stds['Action stds'][i]) + action_means_stds['Action means'][i] 

    # Clip thrust to be between 0 and 60000
    action[3] = np.clip(action[3], 0, 60000)

    return action


def get_action(policy, drone_reader, opti_reader):
    global t_prev, x_prev, z_prev, qx_prev, qy_prev, qz_prev, qw_prev

    time.sleep(3.0)
    print('#############################################################################################')

    start_time = time.time()

    while (time.time() - start_time < 25.0):

        # Get state in real-time from OptiTrack and drone
        drone_reader.lock.acquire()
        try:
            drone_data = drone_reader.value
            # print('Drone lock acquired, drone data: ', drone_data)
        finally:
            drone_reader.lock.release()
            time.sleep(0.01)

        opti_reader.lock.acquire()
        try:
            opti_data = opti_reader.value
            # print('Opti lock acquired, opti data: ', opti_data)
        finally:
            opti_reader.lock.release()
            time.sleep(0.01)

        # print('Drone data: ', drone_data)
        # print('Opti data: ', opti_data)

        # Parse data
        # drone_data is in the form [m1, m2, m3, m4, vbat]
        # opti_data is in the form  [id, x,  y,  z,  qx, qy, qz, qw]
        y    = opti_data[2]
        qx   = opti_data[4]
        qy   = opti_data[5]
        qz   = opti_data[6]
        qw   = opti_data[7]
        m1   = drone_data[0]
        m2   = drone_data[1]
        m3   = drone_data[2]
        m4   = drone_data[3]
        vbat = drone_data[4]
        

        # Calculate velocities and angular velocities
        timestep = time.time() - t_prev
        vx       = abs(opti_data[1] - x_prev) / timestep
        vz       = abs(opti_data[3] - z_prev) / timestep
        omega_qx = abs(qx - qx_prev) / timestep
        omega_qy = abs(qy - qy_prev) / timestep
        omega_qz = abs(qz - qz_prev) / timestep
        omega_qw = abs(qw - qw_prev) / timestep

        # Update 'previous' values
        t_prev  = time.time()
        x_prev  = opti_data[1]
        z_prev  = opti_data[3]
        qx_prev = qx
        qy_prev = qy
        qz_prev = qz
        qw_prev = qw


        # Transform state to normalised state and clip
        state = [y, vx, vz, qx, qy, qz, qw, omega_qx, omega_qy, omega_qz, omega_qw, m1, m2, m3, m4, vbat]
        state = normalise_state(state)

        # Get action from policy
        action = policy(torch.tensor([y, vx, vz, qx, qy, qz, qw, omega_qx, omega_qy, omega_qz, omega_qw, m1, m2, m3, m4, vbat], device='cuda'))

        # Transform action to drone command
        action = action_to_drone_command(action)

        sac_reader.read_data(action)

        print('Action: ', action)

    sac_reader.read_data([0, 0, 0, 0])
    print('Done!')




# Initialise thread data reader objects
# Used for preventing multiple threads from reading/writing at the same time
drone_reader = DataReader()
opti_reader  = DataReader()
sac_reader   = DataReader()


if __name__ == '__main__':

    # Initialise threads; one for drone data and control, one for OptiTrack data, one for policy network

    drone_thread = threading.Thread(target=fly_drone)
    opti_thread = threading.Thread(target=natnet_main)
    policy_thread = threading.Thread(target=get_action, args=(policy,drone_reader,opti_reader,))

    drone_thread.start()
    opti_thread.start()
    policy_thread.start()


