import numpy as np
import mujoco
from termcolor import colored
import rospy
from visualization_msgs.msg import Marker
import time
import sys, os
# Add the path to the utils
sys.path.append(os.path.join(os.path.dirname(__file__)))

from robot import Robot
from interface import SimulatedRobot
from geometry_msgs.msg import Point

def visualize_in_rviz(robot, end_point_traj, end_point_traj_pub):
    
    ee_pos = robot.read_ee_pos(joint_name='joint5')
    
    pts = Point()
    
    pts.x = ee_pos[0]
    pts.y = ee_pos[1]
    pts.z = ee_pos[2]

    end_point_traj.points.append(pts)
    end_point_traj_pub.publish(end_point_traj)
    
def initialize_simulator(Hz = 100.0): 
    world, data = load_world()
    robot = SimulatedRobot(world, data)
    world.opt.timestep = 1.0 / Hz # Control the simulation speed, lower is faster
    data.qpos[:3] = 0.0
    mujoco.mj_forward(world, data)
    print(colored("Simulator Initialized!", 'green'))
    return robot, world, data
    
def load_world(world_path = '/root/lerobot_ros/catkin_ws/src/lerobot_ros/scripts/low_cost_robot/scene.xml'):
        
    # Load world and data
    world = mujoco.MjModel.from_xml_path(world_path)
    data = mujoco.MjData(world)
    print(colored(f"World is loaded from {world_path}", 'green'))
    return world, data
    
def calculate_target_rotation():
    z_axis = np.array([0, 0, 1], dtype=np.float64)  # Gripper z-axis aligned with world z-axis
    x_axis = np.array([1, 0, 0], dtype=np.float64)  # Arbitrary x-axis direction (adjustable)
    
    # Ensure orthogonality
    y_axis = np.cross(z_axis, x_axis)
    y_axis /= np.linalg.norm(y_axis)
    x_axis = np.cross(y_axis, z_axis)
    x_axis /= np.linalg.norm(x_axis)
    
    # Construct rotation matrix
    ee_target_rot = np.column_stack((x_axis, y_axis, z_axis))  # 3x3 rotation matrix
    
    return ee_target_rot

def rotation_matrix_to_quaternion(rot_matrix):
    
    quat = np.zeros(4)
    mujoco.mju_mat2Quat(quat, rot_matrix.flatten())
    return quat

def fix_joint_angle():
    
    rot = calculate_target_rotation()
    return rot.flatten()

def create_marker_traj(ns = "joint6_trajectory"):
    
    trajectory_marker = Marker()
    trajectory_marker.header.frame_id = "world"  # Replace with your Fixed Frame
    trajectory_marker.header.stamp = rospy.Time.now()
    trajectory_marker.ns = ns
    trajectory_marker.id = 0
    trajectory_marker.type = Marker.POINTS  # Type for trajectory
    trajectory_marker.action = Marker.ADD
    trajectory_marker.scale.x = 0.001  # Line width
    trajectory_marker.scale.y = 0.001  # Line width
    trajectory_marker.scale.z = 0.001  # Line width
    trajectory_marker.color.a = 1.0  # Transparency
    trajectory_marker.color.r = 1.0  # Red
    trajectory_marker.color.g = 0.0
    trajectory_marker.color.b = 0.0
    trajectory_marker.pose.orientation.w = 1.0
    
    return trajectory_marker
    

def degree2pwm(degree: np.ndarray) -> np.ndarray:
  '''
  origin_pwm은 모터 기준 0도에 해당하는 pwm 값
  degree는 0~45도 사이의 각도
  '''
  if type(degree) == list:
    degree = np.array(degree)
    
  origin_pwm = 2048
  pwm = origin_pwm + degree * 500 / 45
  
  return pwm

def radian2pwm(radian: np.ndarray) -> np.ndarray:
  
  degree = radian * 180 / np.pi
  pwm = degree2pwm(degree)
  
  return pwm

def pwm2degree(pwm: np.ndarray) -> np.ndarray: 
    '''
    origin_pwm은 모터 기준 0도에 해당하는 pwm 값
    degree는 0~45도 사이의 각도
    '''
    if type(pwm) == list:
        pwm = np.array(pwm)
        
    origin_pwm = 2048
    degree = (pwm - origin_pwm) * 45 / 500
    
    return degree

def pwm2radian(pwm: np.ndarray) -> np.ndarray:
  
  degree = pwm2degree(pwm)
  radian = degree * np.pi / 180
  
  return radian

def clock(step_start, m):
    time_until_next_step = m.opt.timestep - (time.time() - step_start)
    if time_until_next_step > 0:
        time.sleep(time_until_next_step)
        return time.time()
    
    else:
        # print(colored("Warning: Simulation is running slower than real time!", 'red'))
        return time.time()
    
# Connect to the real robot
def initialize_real_robot(world, device_name):
    robot_real = Robot(device_name=device_name)
    print(colored("Real Robot Connected!", 'green'))

    # m = mujoco.MjModel.from_xml_path('low_cost_robot/scene.xml')

    # Set the robot to initial position
    robot_real._set_position_control()
    robot_real._enable_torque()

    ### Initialize the robot_real position
    qpos0 = np.array(robot_real.read_position())
    init_pos = radian2pwm(np.array([0, 0, 0, 0]))
    smooth_mover = np.linspace(qpos0, init_pos, 1000)

    step_start = time.time()
    for revert_pos in smooth_mover:
        robot_real.set_goal_pos([int(p) for p in revert_pos])
        step_start = clock(step_start, world)

    print(colored(f"robot_real is initialized to {pwm2radian(init_pos)}, current position: {pwm2radian(robot_real.read_position())}", 'green'))
    # robot_real._disable_torque()
    
    return robot_real
