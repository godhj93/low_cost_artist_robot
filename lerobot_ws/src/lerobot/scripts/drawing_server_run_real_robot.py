#!/usr/bin/python3

from __future__ import print_function

from lerobot.srv import DrawingRequest, DrawingRequestResponse, DrawingCompleted, DrawingCompletedResponse
import rospy
from geometry_msgs.msg import Point
from visualization_msgs.msg import Marker

from termcolor import colored
import sys
import os
import numpy as np
# Add the path to the utils
sys.path.append(os.path.join(os.path.dirname(__file__)))
from utils.robot_utils import visualize_in_rviz, initialize_simulator, create_marker_traj, initialize_real_robot, radian2pwm
import mujoco.viewer
import mujoco
import pandas as pd
import time
from queue import Queue
import imageio

# 작업 상태 변수
task_done = False
current_task = None
DRAWING_Z = 0.054
DRAWING_Z_COR = 0.002
MOVING_Z = 0.08 # This should be same as 'Z_MAX_BOUND'

# DRAWING_POINT_DIST = 0.0003 # for general things
DRAWING_POINT_DIST = 0.002 # 1cm
INTERPOLATING_NUM = 10

## Camera
import enum

class Resolution(enum.Enum):
  SD = (480, 640)
  HD = (720, 1280)
  UHD = (2160, 3840)


def quartic(t: float) -> float:
  return 0 if abs(t) > 1 else (1 - t**2) ** 2


def blend_coef(t: float, duration: float, std: float) -> float:
  normalised_time = 2 * t / duration - 1
  return quartic(normalised_time / std)


def unit_smooth(normalised_time: float) -> float:
  return 1 - np.cos(normalised_time * 2 * np.pi)


def azimuth(
    time: float, duration: float, total_rotation: float, offset: float
) -> float:
  return offset + unit_smooth(time / duration) * total_rotation

res = Resolution.SD
fps = 60
duration = 1000.0
ctrl_rate = 2
ctrl_std = 0.05
total_rot = 180
blend_std = .8
##

def handle_drawing_request(req):
    
    global current_task, task_done
    
    rospy.loginfo(f"Received {len(req.points)} points to draw.")
    current_task = [Point(x=pt.x, y=pt.y, z=pt.z) for pt in req.points]
    task_done = False 
    return DrawingRequestResponse(success=True)

def handle_drawing_complete(req):
    
    global task_done
    if task_done:
        return DrawingCompletedResponse(success=True)
    
    else:
        return DrawingCompletedResponse(success=False)
   
def process_task():
    
    global FILE_NAME_INDEX
    
    global task_done, current_task, initialized_flag
    
    if current_task is not None:
        rospy.loginfo("Processing the task.")
        
        drawing_true = {
            'x': [],
            'y': [],
            'z': []
        }
        
        drawing_result = {
            'x': [],
            'y': [],
            'z': []
        }
        
        
        for idx, pt in enumerate(current_task):
            
            for i in range(10):

                target_radian = robot.inverse_kinematics_5dof_pen_vertical(
                        ee_target_pos = np.array([pt.x, pt.y, pt.z]),
                        body_name     = 'link5',
                        rate          = 1.0
                    )

                
                # if initialized_flag == False:
                #     smooth_mover = np.linspace(current_pwm, target_pwm, 1000)
                #     print("initalized")
                #     initialized_flag = True
                # else:
                #     if (robot.target_ee_position[-1] < (DRAWING_Z + DRAWING_Z_COR) * 1.2 and np.sqrt(pow(robot.target_ee_position[0]-current_ee_position[0],2) + pow(robot.target_ee_position[1]-current_ee_position[1],2)) < 1e-4) and robot.target_ee_position[-1] > current_ee_position[-1]:
                #         smooth_mover = np.linspace(current_pwm, target_pwm, 1000)
                #         print(colored(f"smooth mover : {1000} (going up)", 'green'))
                #     elif robot.target_ee_position[-1] > (DRAWING_Z + DRAWING_Z_COR) * 1.1:
                #         smooth_mover = np.linspace(current_pwm, target_pwm, 100)
                #         z_up_flag = True
                #         z_going_down_iter = 0
                #         print(colored(f"smooth mover : {100} (moving in the air)", 'green'))
                #     elif z_up_flag:
                #         smooth_mover = np.linspace(current_pwm, target_pwm, 100)
                #         z_up_flag = False
                #         z_going_down_iter = 0
                #         print(colored(f"smooth mover : {1000} (going down)", 'blue'))
                #     elif z_going_down_iter < 5:
                #         z_going_down_iter += 1
                #         smooth_mover = np.linspace(current_pwm, target_pwm, 1000)
                #         z_up_flag = False
                #         print(colored(f"smooth mover : {1000} (starting moving on the ground)", 'blue'))
                #     else:
                #         smooth_mover = np.linspace(current_pwm, target_pwm, INTERPOLATING_NUM)
                #         print(colored(f"smooth mover : {INTERPOLATING_NUM}", 'red'))
                #         # smooth_mover = np.array([target_pwm])
                #         # print(colored(f"smooth mover : {1}", 'red'))

                # step_start = time.time()
                # for pwm in smooth_mover:
                #     real_robot.set_goal_pos([int(p) for p in pwm])
                #     step_start = clock(step_start, world)
                    
                
                current_pwm = real_robot.read_position()

                target_pwm = radian2pwm(np.array(target_radian[:4]))
                
                real_robot.set_goal_pos([int(p) for p in target_pwm])
                viewer.sync()
                
                error = np.linalg.norm(robot.read_ee_pos(joint_name='joint5') - np.array([pt.x, pt.y, pt.z]))
                if error < 0.003:
                    print(colored(f"Point {idx+1}/{len(current_task)} reached.", 'green'))
                    break
                    
                
            
            if pt.z <= 0.1:
                
                ee_x, ee_y, ee_z = visualize_in_rviz(robot, robot.read_ee_pos(joint_name='joint5'), end_point_traj, end_point_traj_pub)
                visualize_in_rviz(robot, np.array([pt.x, pt.y, pt.z]), end_point_true, end_point_true_pub)
                
                end_point_true.points.append(pt)
                end_point_true_pub.publish(end_point_true)
                
                drawing_true['x'].append(pt.x)
                drawing_true['y'].append(pt.y)
                drawing_true['z'].append(pt.z)
                
                drawing_result['x'].append(ee_x)
                drawing_result['y'].append(ee_y)
                drawing_result['z'].append(ee_z)
                viewer.sync()
            if rospy.is_shutdown():
                rospy.logerr("ROS shutdown detected.")
                break
            
            rospy.loginfo(f"Drawing point {idx+1}/{len(current_task)}")        

        viewer.sync()
        # Make directory
        if not os.path.exists('trajectory_result'):
            os.makedirs('trajectory_result', exist_ok=True)
        
        # Convert the trajectory to pandas dataframe
        traj_df = pd.DataFrame(drawing_true)
        traj_df.to_csv(f'trajectory_result/trajectory_{FILE_NAME_INDEX}_true.csv', index=False)
        
        traj_df = pd.DataFrame(drawing_result)
        traj_df.to_csv(f'trajectory_result/trajectory_{FILE_NAME_INDEX}_result.csv', index=False)
        
        FILE_NAME_INDEX += 1
        
        rospy.logwarn("Task completed.")
        task_done = True
        current_task = None
        
        imageio.mimwrite(f'video_{FILE_NAME_INDEX}.mp4', frames, fps=fps)
        return FILE_NAME_INDEX
        
if __name__ == "__main__":
    
    # Initialize the simulator
    robot, world, data = initialize_simulator(Hz = 100)
    
    # Set simulation timestep
    world.opt.timestep = 1.0 / 500  # Simulation timestep in seconds (500 Hz)

    # Initialize the robot position
    data.qpos[0] = 0.0
    data.qpos[1] = 0.0
    data.qpos[2] = 0.0
    mujoco.mj_forward(world, data)

    # Desired real-time frame rate
    frame_duration = 1.0 / 300
    
    # Initialize real robot
    real_robot = initialize_real_robot(world, device_name='/dev/ttyACM0')
    real_robot._set_position_control()
    real_robot._enable_torque()
    
    real_robot.set_pid_gain(1, 1000, 100, 1000)
    real_robot.set_pid_gain(2, 1500, 1000, 400)
    real_robot.set_pid_gain(3, 2000, 1000, 400)
    real_robot.set_pid_gain(4, 2000, 1000, 400)
    
    for id in [1, 2, 3, 4]:
        print(f"Dynamixel {id}: ")
        real_robot.read_gain(id, 84, "P")
        real_robot.read_gain(id, 82, "I")
        real_robot.read_gain(id, 80, "D")
        print()
        
    # Initialize target position
    robot.target_ee_position = np.array([0.0, 0.3, 0.15])
    initialized_flag = False
    
    # Initialize the ROS node
    rospy.init_node("drawing_server")
    
    # Define the service
    rospy.Service("drawing_request", DrawingRequest, handle_drawing_request)
    rospy.Service("drawing_completed", DrawingCompleted, handle_drawing_complete)
    
    end_point_traj = create_marker_traj(color='green')
    end_point_traj_pub = rospy.Publisher("end_point_traj", Marker, queue_size=10)
    
    end_point_true = create_marker_traj(ns="end_point_true", color='white')
    end_point_true_pub = rospy.Publisher("end_point_true", Marker, queue_size=10)
    
    rospy.loginfo("Drawing server is ready to receive requests.")

    global FILE_NAME_INDEX
    FILE_NAME_INDEX = 0
    with mujoco.viewer.launch_passive(world, data) as viewer:
        try:
            while viewer.is_running():
                
                process_task()
                end_point_traj.points = []
                end_point_true.points = []  
                # Check the keyboard interrupt
                if rospy.is_shutdown():
                    rospy.logerr("ROS shutdown detected.")
                    break
                    
        except rospy.ROSInterruptException:
            print("ROS interrupt received.")
        finally:
            print("Exiting program.")