#!/usr/bin/python3

from __future__ import print_function

from lerobot_ros.srv import DrawingRequest, DrawingRequestResponse, DrawingCompleted, DrawingCompletedResponse
import rospy
from geometry_msgs.msg import Point
from visualization_msgs.msg import Marker

from termcolor import colored
import sys
import os
import numpy as np
# Add the path to the utils
sys.path.append(os.path.join(os.path.dirname(__file__)))
from utils.robot_utils import fix_joint_angle, create_marker_traj, initialize_simulator, visualize_in_rviz
import mujoco.viewer
import mujoco
import pandas as pd
# 작업 상태 변수
task_done = False
current_task = None

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
    
    global task_done, current_task
    
    if current_task is not None:
        rospy.loginfo("Processing the task.")
        
        drawing_results = {
            'x': [],
            'y': [],
            'z': []
        }
        for idx, pt in enumerate(current_task):
            
            robot.target_ee_position = np.array([pt.x, pt.y, pt.z])
            
            robot.inverse_kinematics_rot_backup_5DOF(
                    ee_target_pos=robot.target_ee_position, 
                    ee_target_rot=fix_joint_angle(), 
                    joint_name='joint5')
            
            visualize_in_rviz(robot, end_point_traj, end_point_traj_pub)
            viewer.sync()
            
            rospy.loginfo(f"Drawing point {idx+1}/{len(current_task)}")        
            
            drawing_results['x'].append(pt.x)
            drawing_results['y'].append(pt.y)
            drawing_results['z'].append(pt.z)
            
            if rospy.is_shutdown():
                rospy.logerr("ROS shutdown detected.")
                break
            
        # Make directory
        if not os.path.exists('trajectory_result'):
            os.makedirs('trajectory_result', exist_ok=True)
        
        # Convert the trajectory to pandas dataframe
        traj_df = pd.DataFrame(drawing_results)
        traj_df.to_csv(f'trajectory_result/trajectory_{FILE_NAME_INDEX}.csv', index=False)
        FILE_NAME_INDEX += 1
        
        rospy.logwarn("Task completed.")
        task_done = True
        current_task = None
        
        return FILE_NAME_INDEX
        
        
if __name__ == "__main__":
    
    # Initialize the simulator
    robot, world, data = initialize_simulator(Hz = 100)
    
    # Initialize the ROS node
    rospy.init_node("drawing_server")
    
    # Define the service
    rospy.Service("drawing_request", DrawingRequest, handle_drawing_request)
    rospy.Service("drawing_completed", DrawingCompleted, handle_drawing_complete)
    
    end_point_traj = create_marker_traj()
    end_point_traj_pub = rospy.Publisher("end_point_traj", Marker, queue_size=10)
    rospy.loginfo("Drawing server is ready to receive requests.")

    global FILE_NAME_INDEX
    FILE_NAME_INDEX = 0
    
    with mujoco.viewer.launch_passive(world, data) as viewer:
        try:
            while viewer.is_running():
                
                process_task()
                
                end_point_traj.points = []
                
                # Check the keyboard interrupt
                if rospy.is_shutdown():
                    rospy.logerr("ROS shutdown detected.")
                    break
                    
        except rospy.ROSInterruptException:
            print("ROS interrupt received.")
        finally:
            print("Exiting program.")