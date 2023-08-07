# crazyflie imports
import logging
import time
import csv
import pandas as pd
import os

import cflib.crtp
from cflib.crazyflie import Crazyflie
from cflib.crazyflie.syncCrazyflie import SyncCrazyflie
from cflib.utils import uri_helper
from cflib.crazyflie.log import LogConfig
from cflib.crazyflie.syncLogger import SyncLogger

import cf_thrust_fns


# python-natnet copyright notice
"""Command-line NatNet client application for testing.

Copyright (c) 2017, Matthew Edwards.  This file is subject to the 3-clause BSD
license, as found in the LICENSE file in the top-level directory of this
distribution and at https://github.com/mje-nz/python_natnet/blob/master/LICENSE.
No part of python_natnet, including this file, may be copied, modified,
propagated, or distributed except according to the terms contained in the
LICENSE file.
"""
#natnet imports
import argparse
import time
import os
import csv

import attr

import natnet

#threading imports
import threading


# Crazyflie initialise radio connection
uri = uri_helper.uri_from_env(default='radio://0/80/2M/E7E7E7E7E7')
# Only output errors from the logging framework
logging.basicConfig(level=logging.ERROR)

stab = 0
motor_signals = [0, 0, 0, 0]

def log_motor_callback(timestamp, data, logconf):
    # print(data)
    global motor_signals
    
    motor_signals[0] = data['motor.m1']
    motor_signals[1] = data['motor.m2']
    motor_signals[2] = data['motor.m3']
    motor_signals[3] = data['motor.m4']

    print('Motor PWM: ', [motor_signals[0], motor_signals[1], motor_signals[2], motor_signals[3]])

    global stab
    stab = data['stabilizer.thrust']

    # global file_extension

    # file_path = './' + 'thrust_data' + '/' + file_extension

    # with open(os.path.join(file_path,
    #         'data.csv'), 'a') as fd:
    #     cwriter = csv.writer(fd)
    #     cwriter.writerow([time.time(), stab, motor_signals[0], motor_signals[1], motor_signals[2], motor_signals[3]]) # time.time() is time since 'epoch' - Jan 1 1970 00:00

    global drone_reader
    drone_reader.read_data(motor_signals)



# CF flight code
def fly_drone():
    
    # folder = 'thrust_data'
    # file_path = './' + folder + '/' + file_extension

    # # Create folder
    # if not os.path.exists(file_path):
    #     os.makedirs(file_path)
    
    # with open(os.path.join(file_path,
    #         'data.csv'), 'a') as fd:
    #     cwriter = csv.writer(fd)
    #     cwriter.writerow(['Time', 'stabilizer.thrust', 'motor.m1', 'motor.m2', 'motor.m3', 'motor.m4']) 
    
    
    # Initialize the low-level drivers
    cflib.crtp.init_drivers()

    lg_motor = LogConfig(name='Data', period_in_ms=10)
    lg_motor.add_variable('stabilizer.thrust', 'float')
    # lg_motor = LogConfig(name='Motors', period_in_ms=10)
    lg_motor.add_variable('motor.m1', 'float')
    lg_motor.add_variable('motor.m2', 'float')
    lg_motor.add_variable('motor.m3', 'float')
    lg_motor.add_variable('motor.m4', 'float')

    

    with SyncCrazyflie(uri, cf=Crazyflie(rw_cache='./cache')) as scf:

        try:
            scf.cf.log.add_config(lg_motor)
            lg_motor.data_received_cb.add_callback(log_motor_callback)


            lg_motor.start()

            # cf_thrust_fns.thrust_ramp(scf)
            cf_thrust_fns.thrust_from_file(scf)
            # cf_thrust_fns.ramp_motors(scf)
            # ramp_motors(scf)
            # cf_thrust_fns.motors_from_file(scf)



            lg_motor.stop()
            
        except KeyboardInterrupt:
            scf.cf.param.set_value('motorPowerSet.enable', 0)
            print('Sending shutdown command')
            

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

        # if markers:
        #     print('Markers')
        #     for m in markers:
        #         print('\t Model {} marker {}: size {:.4f}mm, pos ({: 5.2f}, {: 5.2f}, {: 5.2f}), '.format(
        #             m.model_id, m.marker_id, 1000*m.size, *m.position
        #         ))
        # print('\t Latency: {:.1f}ms (system {:.1f}ms, transit {:.1f}ms, processing {:.2f}ms)'.format(
        #     1000*timing.latency, 1000*timing.system_latency, 1000*timing.transit_latency,
        #     1000*timing.processing_latency
        # ))

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


    # folder = 'mocap_data'
    # file_path = './' + folder + '/' + file_extension

    # # Create experiment folder
    # if not os.path.exists(file_path):
    #     os.makedirs(file_path)

    
    # with open(os.path.join(file_path,
    #         'data.csv'), 'a') as fd:
    #     cwriter = csv.writer(fd)
    #     cwriter.writerow(['Time', 'ID', 'Pos x', 'Pos y', 'Pos z', 'Rot 1', 'Rot 2', 'Rot 3', 'Rot 4']) # time.time() is time since 'epoch' - Jan 1 1970 00:00


    try:
        app = ClientApp.connect('fake' if args.fake else args.server, args.rate, args.quiet)
        app.run()
    except natnet.DiscoveryError as e:
        print('Error:', e)



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
            print('Data in: ', self.value)
        finally:
            logging.debug('Released a lock')
            self.lock.release()
        
        # time.sleep(0.01)


def write_to_csv(file_path, drone_data, opti_data):
    with open(os.path.join(file_path,
            'data.csv'), 'a') as fd:
        cwriter = csv.writer(fd)
        # print('To csv: ', [time.time()], drone_data, opti_data)
        cwriter.writerow([time.time()] + drone_data + opti_data) # time.time() is time since 'epoch' - Jan 1 1970 00:00
        # print(drone_data)
        # print(opti_data)




def write_drone_opti(drone_reader, opti_reader, file_path):

    while True:

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

        write_to_csv(file_path, drone_data, opti_data)
        


drone_reader = DataReader()
opti_reader = DataReader()

if __name__ == '__main__':


    folder = 'thrust_mocap_data'
    file_extension = str(time.ctime().replace(' ', '-'))
    file_path = './' + folder + '/' + file_extension

    # Create folder
    if not os.path.exists(file_path):
        os.makedirs(file_path)

    with open(os.path.join(file_path,
        'data.csv'), 'a') as fd:
        cwriter = csv.writer(fd)
        cwriter.writerow(['Time', 'motor.m1', 'motor.m2', 'motor.m3', 'motor.m4', 'RB ID', 'Pos x', 'Pos y', 'Pos z', 'Quat w', 'Quat x', 'Quat y', 'Quat z']) 
    
    

    drone_thread = threading.Thread(target=fly_drone)
    opti_thread = threading.Thread(target=natnet_main)
    main_thread = threading.Thread(target=write_drone_opti, args=(drone_reader, opti_reader, file_path))

    drone_thread.start()
    opti_thread.start()
    main_thread.start()

