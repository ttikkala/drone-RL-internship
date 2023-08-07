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

# import data_threading_OLD

import atexit

# uri = uri_helper.uri_from_env(default='radio://0/80/2M/E7E7E7E7E7')
uri = uri_helper.uri_from_env(default='radio://0/100/2M/E7E7E7E7E7')
# Only output errors from the logging framework
logging.basicConfig(level=logging.ERROR)

thrust_file = '~/.config/cfclient/logdata/20230626T11-17-16/Motors-20230626T11-17-20.csv'
data = pd.read_csv(thrust_file)

m1_input = data['motor.m1'].tolist()
m2_input = data['motor.m2'].tolist()
m3_input = data['motor.m3'].tolist()
m4_input = data['motor.m4'].tolist()

stab_thrust_input = data['stabilizer.thrust'].tolist()

stab = 0
motor_signals = [0, 0, 0, 0]

file_extension = str(time.ctime().replace(' ', '_'))


def log_motor_callback(timestamp, data, logconf):
    # print(data)
    global motor_signals
    motor_signals[0] = data['motor.m1']
    motor_signals[1] = data['motor.m2']
    motor_signals[2] = data['motor.m3']
    motor_signals[3] = data['motor.m4']

    global stab
    stab = data['stabilizer.thrust']

    global file_extension

    file_path = './' + 'thrust_data' + '/' + file_extension

    with open(os.path.join(file_path,
            'data.csv'), 'a') as fd:
        cwriter = csv.writer(fd)
        cwriter.writerow([time.time(), stab, motor_signals[0], motor_signals[1], motor_signals[2], motor_signals[3]]) # time.time() is time since 'epoch' - Jan 1 1970 00:00

    # drone_data_reader = data_threading.drone_reader
    # drone_data_reader.read_data(motor_signals)



def thrust_ramp(scf):

    thrust_mult = 1
    thrust_step = 500
    thrust = 10001
    pitch = 0
    roll = 0
    yawrate = 0

    # Unlock startup thrust protection
    scf.cf.commander.send_setpoint(0, 0, 0, 0)

    while thrust >= 10001:
        scf.cf.commander.send_setpoint(roll, pitch, yawrate, thrust)
        time.sleep(0.1)
        if thrust >= 30000:
            thrust_mult = -1
        thrust += thrust_step * thrust_mult
    scf.cf.commander.send_setpoint(0, 0, 0, 0)
    # Make sure that the last packet leaves before the link is closed
    # since the message queue is not flushed before closing
    time.sleep(0.1)
    scf.cf.close_link()


def thrust_from_file(scf):

    scf.cf.commander.send_setpoint(0, 0, 0, 0)

    for thrust in stab_thrust_input:
        time.sleep(0.1)
        scf.cf.commander.send_setpoint(0, 0, 0, thrust) # roll, pitch, yawrate, thrust

    scf.cf.commander.send_setpoint(0, 0, 0, 0)
    time.sleep(0.1)
    scf.cf.close_link()


def ramp_motors(scf):

        thrust_mult = 1
        thrust_step = 500
        time_step = 0.1
        thrust = 5000
        pitch = 0
        roll = 0
        yawrate = 0

        scf.cf.commander.send_setpoint(0, 0, 0, 0)

        # scf.cf.param.set_value('motor.batCompensation', 0)
        scf.cf.param.set_value('motorPowerSet.m1', 0)
        scf.cf.param.set_value('motorPowerSet.m2', 0)
        scf.cf.param.set_value('motorPowerSet.m3', 0)
        scf.cf.param.set_value('motorPowerSet.m4', 0)
        scf.cf.param.set_value('motorPowerSet.enable', 2)
        scf.cf.param.set_value('system.forceArm', 1)

        while scf.cf.is_connected: #thrust >= 0:
            thrust += thrust_step * thrust_mult
            if thrust >= 13000 or thrust < 0:
                thrust_mult *= -1
                thrust += thrust_step * thrust_mult
            print(thrust)
            # scf.cf._cf.commander.send_setpoint(roll, pitch, yawrate, thrust)
            # localization.send_emergency_stop_watchdog()
            scf.cf.param.set_value('motorPowerSet.m1', str(thrust))
            scf.cf.param.set_value('motorPowerSet.m2', str(thrust))
            scf.cf.param.set_value('motorPowerSet.m3', str(thrust))
            scf.cf.param.set_value('motorPowerSet.m4', str(thrust))
            time.sleep(time_step)

            # scf.cf.commander.send_setpoint(0, 0, 0, 0)
            # time.sleep(10)

        # scf.cf.commander.send_setpoint(0, 0, 0, 0)
        # Make sure that the last packet leaves before the link is closed
        # since the message queue is not flushed before closing
        time.sleep(0.1)
        scf.cf.close_link()


def motors_from_file(scf):

    data = pd.read_csv(thrust_file)

    m1_input = data['motor.m1'].to_numpy()
    m2_input = data['motor.m2'].to_numpy()
    m3_input = data['motor.m3'].to_numpy()
    m4_input = data['motor.m4'].to_numpy()

    # stab_thrust_input = data['stabilizer.thrust'].tolist()

    motor_inputs = [m1_input, m2_input, m3_input, m4_input]
     
    scf.cf.param.set_value('motorPowerSet.m1', 0)
    scf.cf.param.set_value('motorPowerSet.m2', 0)
    scf.cf.param.set_value('motorPowerSet.m3', 0)
    scf.cf.param.set_value('motorPowerSet.m4', 0)
    scf.cf.param.set_value('motorPowerSet.enable', 2)
    scf.cf.param.set_value('system.forceArm', 1)

    while scf.cf.is_connected:
        for idx in range(len(motor_inputs[0])):
            time.sleep(0.1)
            scf.cf.param.set_value('motorPowerSet.m1', str(motor_inputs[0][idx]))
            # print('Motor inputs 1: ', [motor_inputs[0][idx], motor_inputs[1][idx], motor_inputs[2][idx], motor_inputs[3][idx]])
            scf.cf.param.set_value('motorPowerSet.m2', str(motor_inputs[1][idx]))
            # print('Motor inputs 2: ', [motor_inputs[0][idx], motor_inputs[1][idx], motor_inputs[2][idx], motor_inputs[3][idx]])
            scf.cf.param.set_value('motorPowerSet.m3', str(motor_inputs[2][idx]))
            # print('Motor inputs 3: ', [motor_inputs[0][idx], motor_inputs[1][idx], motor_inputs[2][idx], motor_inputs[3][idx]])
            scf.cf.param.set_value('motorPowerSet.m4', str(motor_inputs[3][idx]))
            print('Motor inputs: ', [motor_inputs[0][idx], motor_inputs[1][idx], motor_inputs[2][idx], motor_inputs[3][idx]])

def hover_auto(scf):

    scf.cf.commander.send_setpoint(0, 0, 0, 0)
    scf.cf.commander.send_hover_setpoint(0, 0, 0, 0.1)
    time.sleep(5)
    scf.cf.commander.send_hover_setpoint(0, 0, 0, 0)


def main():
    
    folder = 'thrust_data'
    file_path = './' + folder + '/' + file_extension

    # Create folder
    if not os.path.exists(file_path):
        os.makedirs(file_path)
    
    with open(os.path.join(file_path,
            'data.csv'), 'a') as fd:
        cwriter = csv.writer(fd)
        cwriter.writerow(['Time', 'stabilizer.thrust', 'motor.m1', 'motor.m2', 'motor.m3', 'motor.m4']) 
    
    
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
            # scf.cf.log.add_config(lg_stab)
            # lg_stab.data_received_cb.add_callback(log_stab_callback)
            scf.cf.log.add_config(lg_motor)
            lg_motor.data_received_cb.add_callback(log_motor_callback)

            lg_motor.start()

            thrust_ramp(scf)
            # thrust_from_file(scf)
            # ramp_motors(scf)
            # motors_from_file(scf)
            # hover_auto(scf)

            lg_motor.stop()
            
        except KeyboardInterrupt:
            scf.cf.param.set_value('motorPowerSet.enable', 0)
            print('Sending shutdown command')
            



if __name__ == '__main__':
    main()






# atexit.register(zero_thrust)


# import signal

# def handle_exit():
#     print('In exit fn')
#     with SyncCrazyflie(uri, cf=Crazyflie(rw_cache='./cache')) as scf:
#         print('Sending shutdown command')
#         scf.cf.commander.send_setpoint(0, 0, 0, 0)



# atexit.register(handle_exit)
# signal.signal(signal.SIGTERM, handle_exit)
# signal.signal(signal.SIGINT, handle_exit)