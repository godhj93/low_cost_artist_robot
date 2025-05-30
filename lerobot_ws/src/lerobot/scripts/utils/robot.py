# source: https://github.com/reedscot/low_cost_robot/blob/main/robot.py
import numpy as np
from dynamixel import Dynamixel, OperatingMode, ReadAttribute
import time
# from dynamixel_sdk import GroupSyncRead, GroupSyncWrite, DXL_LOBYTE, DXL_HIBYTE, DXL_LOWORD, DXL_HIWORD
from dynamixel_sdk import *
from enum import Enum, auto
from typing import Union

class MotorControlType(Enum):
    PWM = auto()
    POSITION_CONTROL = auto()
    VELOCITY_CONTROL = auto()
    DISABLED = auto()
    UNKNOWN = auto()



class Robot:
    def __init__(self, device_name: str, baudrate=1_000_000, servo_ids=[1, 2, 3, 4]):
        self.servo_ids = servo_ids
        self.dynamixel = Dynamixel.Config(baudrate=baudrate, device_name=device_name).instantiate()
        self.torque_reader = GroupSyncRead(
            self.dynamixel.portHandler,
            self.dynamixel.packetHandler,
            ReadAttribute.TORQUE.value,
            1)
        for id in self.servo_ids:
            self.torque_reader.addParam(id)

        self.position_reader = GroupSyncRead(
            self.dynamixel.portHandler,
            self.dynamixel.packetHandler,
            ReadAttribute.POSITION.value,
            4)
        for id in self.servo_ids:
            self.position_reader.addParam(id)

        self.velocity_reader = GroupSyncRead(
            self.dynamixel.portHandler,
            self.dynamixel.packetHandler,
            ReadAttribute.VELOCITY.value,
            4)
        for id in self.servo_ids:
            self.velocity_reader.addParam(id)

        self.pos_writer = GroupSyncWrite(
            self.dynamixel.portHandler,
            self.dynamixel.packetHandler,
            self.dynamixel.ADDR_GOAL_POSITION,
            4)
        for id in self.servo_ids:
            self.pos_writer.addParam(id, [2048])

        self.pwm_writer = GroupSyncWrite(
            self.dynamixel.portHandler,
            self.dynamixel.packetHandler,
            self.dynamixel.ADDR_GOAL_PWM,
            2)
        for id in self.servo_ids:
            self.pwm_writer.addParam(id, [2048])
        
        self.vel_writer = GroupSyncWrite(
            self.dynamixel.portHandler,
            self.dynamixel.packetHandler,
            self.dynamixel.ADDR_GOAL_VELOCITY,
            4)
        for id in self.servo_ids:
            self.vel_writer.addParam(id, [2048])

        # for id in self.servo_ids:
        #     # if id == 2:
        #     #     self.set_pid_gain(id, 1000, 100, 0)
        #     #     self.read_gain(id, 84, "P")
        #     #     self.read_gain(id, 82, "I")
        #     #     self.read_gain(id, 80, "D")
        #     if id == 3: 
        #         self.set_pid_gain(id, 1000, 100, 600)
        #         self.read_gain(id, 84, "P")
        #         self.read_gain(id, 82, "I")
        #         self.read_gain(id, 80, "D")
        #     if id == 4: 
        #         self.set_pid_gain(id, 1000, 100, 600)
        #         self.read_gain(id, 84, "P")
        #         self.read_gain(id, 82, "I")
        #         self.read_gain(id, 80, "D")

        self._disable_torque()
        self.motor_control_state = MotorControlType.DISABLED

        # for id in self.servo_ids:
        #     # if id == 2:
        #     #     self.set_pid_gain(id, 1000, 100, 0)
        #     #     self.read_gain(id, 84, "P")
        #     #     self.read_gain(id, 82, "I")
        #     #     self.read_gain(id, 80, "D")
        #     if id == 3: 
        #         print("HERE")
        #         self.read_gain(id, 84, "P")
        #         self.read_gain(id, 82, "I")
        #         self.read_gain(id, 80, "D")
        #     if id == 4: 
        #         print("HERE2")
        #         self.read_gain(id, 84, "P")
        #         self.read_gain(id, 82, "I")
        #         self.read_gain(id, 80, "D")

    def read_position(self, tries=5):
        """
        Reads the joint positions of the robot. 2048 is the center position. 0 and 4096 are 180 degrees in each direction.
        :param tries: maximum number of tries to read the position
        :return: list of joint positions in range [0, 4096]
        """
        result = self.position_reader.txRxPacket()
        if result != 0:
            if tries > 0:
                return self.read_position(tries=tries - 1)
            else:
                print(f'failed to read position!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
        positions = []
        for id in self.servo_ids:
            position = self.position_reader.getData(id, ReadAttribute.POSITION.value, 4)
            if position > 2 ** 31:
                position -= 2 ** 32
            positions.append(position)
        return positions
    
    def read_torque_onoff(self):
        """
        Reads the torque on off of the robot.
        """
        self.torque_reader.txRxPacket()
        torque = []
        for id in self.servo_ids:
            t = self.torque_reader.getData(id, ReadAttribute.TORQUE.value, 1)
            if t > 2 ** 31:
                t -= 2 ** 32
            torque.append(t)
        return torque
    
    def read_velocity(self, tries=5):
        """
        Reads the joint velocities of the robot.
        :return: list of joint velocities,
        """
        result = self.velocity_reader.txRxPacket()
        if result != 0:
            if tries > 0:
                return self.read_velocity(tries=tries - 1)
            else:
                print(f'failed to read velocity!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
        
        velocties = []
        for id in self.servo_ids:
            velocity = self.velocity_reader.getData(id, ReadAttribute.VELOCITY.value, 4)
            if velocity > 2 ** 31:
                velocity -= 2 ** 32
            velocties.append(velocity)
        return velocties

    def set_goal_pos(self, action):
        """

        :param action: list or numpy array of target joint positions in range [0, 4096]
        """
        if not self.motor_control_state is MotorControlType.POSITION_CONTROL:
            self._set_position_control()
        for i, motor_id in enumerate(self.servo_ids):
            data_write = [DXL_LOBYTE(DXL_LOWORD(action[i])),
                          DXL_HIBYTE(DXL_LOWORD(action[i])),
                          DXL_LOBYTE(DXL_HIWORD(action[i])),
                          DXL_HIBYTE(DXL_HIWORD(action[i]))]
            self.pos_writer.changeParam(motor_id, data_write)

        self.pos_writer.txPacket()

    def set_pwm(self, action):
        """
        Sets the pwm values for the servos.
        :param action: list or numpy array of pwm values in range [0, 885]
        """
        if not self.motor_control_state is MotorControlType.PWM:
            self._set_pwm_control()
        for i, motor_id in enumerate(self.servo_ids):
            data_write = [DXL_LOBYTE(DXL_LOWORD(action[i])),
                          DXL_HIBYTE(DXL_LOWORD(action[i])),
                          ]
            self.pwm_writer.changeParam(motor_id, data_write)

        self.pwm_writer.txPacket()

    def set_velocity(self, action):
        """
        Sets the velocity values for the servos.
        :param action: list or numpy array of pwm values in range [-445, 445]
        """
        if not self.motor_control_state is MotorControlType.VELOCITY_CONTROL:
            self._set_velocity_control()
        for i, motor_id in enumerate(self.servo_ids):
            data_write = [DXL_LOBYTE(DXL_LOWORD(action[i])),
                          DXL_HIBYTE(DXL_LOWORD(action[i])),
                          DXL_LOBYTE(DXL_HIWORD(action[i])),
                          DXL_HIBYTE(DXL_HIWORD(action[i]))]
            self.vel_writer.changeParam(motor_id, data_write)

        self.vel_writer.txPacket()

    def set_trigger_torque(self):
        """
        Sets a constant torque torque for the last servo in the chain. This is useful for the trigger of the leader arm
        """
        self.dynamixel._enable_torque(self.servo_ids[-1])
        self.dynamixel.set_pwm_value(self.servo_ids[-1], 200)

    def limit_pwm(self, limit: Union[int, list, np.ndarray]):
        """
        Limits the pwm values for the servos in for position control
        @param limit: 0 ~ 885
        @return:
        """
        if isinstance(limit, int):
            limits = [limit, ] * 6
        else:
            limits = limit
        self._disable_torque()
        for motor_id, limit in zip(self.servo_ids, limits):
            self.dynamixel.set_pwm_limit(motor_id, limit)
        self._enable_torque()

    def set_pid_gain(self, dxl_id, p_gain, i_gain, d_gain):
        # P 게인 설정
        dxl_comm_result, dxl_error = self.dynamixel.packetHandler.write2ByteTxRx(
            self.dynamixel.portHandler, dxl_id, 84, p_gain
        )
        if dxl_comm_result != COMM_SUCCESS:
            print(f"Failed to set P Gain: {self.dynamixel.packetHandler.getTxRxResult(dxl_comm_result)}")
        elif dxl_error != 0:
            print(f"Error occurred: {self.dynamixel.packetHandler.getRxPacketError(dxl_error)}")
        
        # I 게인 설정
        dxl_comm_result, dxl_error = self.dynamixel.packetHandler.write2ByteTxRx(
            self.dynamixel.portHandler, dxl_id, 82, i_gain
        )
        if dxl_comm_result != COMM_SUCCESS:
            print(f"Failed to set I Gain: {self.dynamixel.packetHandler.getTxRxResult(dxl_comm_result)}")
        elif dxl_error != 0:
            print(f"Error occurred: {self.dynamixel.packetHandler.getRxPacketError(dxl_error)}")

        # D 게인 설정
        dxl_comm_result, dxl_error = self.dynamixel.packetHandler.write2ByteTxRx(
            self.dynamixel.portHandler, dxl_id, 80, d_gain
        )
        if dxl_comm_result != COMM_SUCCESS:
            print(f"Failed to set D Gain: {self.dynamixel.packetHandler.getTxRxResult(dxl_comm_result)}")
        elif dxl_error != 0:
            print(f"Error occurred: {self.dynamixel.packetHandler.getRxPacketError(dxl_error)}")
            
    def read_gain(self, dxl_id, address, name):
        dxl_gain, dxl_comm_result, dxl_error = self.dynamixel.packetHandler.read2ByteTxRx(
            self.dynamixel.portHandler, dxl_id, address
        )
        if dxl_comm_result != COMM_SUCCESS:
            print(f"Failed to read {name} Gain: {self.dynamixel.packetHandler.getTxRxResult(dxl_comm_result)}")
        elif dxl_error != 0:
            print(f"Error occurred while reading {name} Gain: {self.dynamixel.packetHandler.getRxPacketError(dxl_error)}")
        else:
            print(f"{name} Gain read successfully: {dxl_gain}")
        return dxl_gain

    def _disable_torque(self):
        print(f'disabling torque for servos {self.servo_ids}')
        for motor_id in self.servo_ids:
            self.dynamixel._disable_torque(motor_id)

    def _enable_torque(self):
        print(f'enabling torque for servos {self.servo_ids}')
        for motor_id in self.servo_ids:
            self.dynamixel._enable_torque(motor_id)

    def _set_pwm_control(self):
        self._disable_torque()
        for motor_id in self.servo_ids:
            self.dynamixel.set_operating_mode(motor_id, OperatingMode.PWM)
        # self._enable_torque()
        self.motor_control_state = MotorControlType.PWM

    def _set_position_control(self):
        # if sum(self.read_torque_onoff()) != 0:
        #     print('DISABLE TORQUE FOR POSITION CONTROL')
        #     self._disable_torque()
        for motor_id in self.servo_ids:
            self.dynamixel.set_operating_mode(motor_id, OperatingMode.POSITION)
        # self._enable_torque()
        self.motor_control_state = MotorControlType.POSITION_CONTROL

    def _set_velocity_control(self):
        # if sum(self.read_torque_onoff()) != 0:
        #     print('DISABLE TORQUE FOR VELOCITY CONTROL')
        #     self._disable_torque()
        for motor_id in self.servo_ids:
            self.dynamixel.set_operating_mode(motor_id, OperatingMode.VELOCITY)
        # self._enable_torque()
        self.motor_control_state = MotorControlType.VELOCITY_CONTROL
        
if __name__ == "__main__":
    robot = Robot(device_name='/dev/ttyACM0')
    robot._disable_torque()
    for _ in range(10000):
        s = time.time()
        pos = robot.read_position()
        elapsed = time.time() - s
        print(f'read took {elapsed} pos {pos}')