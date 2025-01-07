import mujoco
import numpy as np
from termcolor import colored

class SimulatedRobot:
    def __init__(self, m, d) -> None:
        """
        :param m: mujoco model
        :param d: mujoco data
        """
        self.m = m
        self.d = d

    def _pos2pwm(self, pos: np.ndarray) -> np.ndarray:
        """
        :param pos: numpy array of joint positions in range [-pi, pi]
        :return: numpy array of pwm values in range [0, 4096]
        """
        return (pos / 3.14 + 1.) * 4096

    def _pwm2pos(self, pwm: np.ndarray) -> np.ndarray:
        """
        :param pwm: numpy array of pwm values in range [0, 4096]
        :return: numpy array of joint positions in range [-pi, pi]
        """
        return (pwm / 2048 - 1) * 3.14 

    def _pwm2norm(self, x: np.ndarray) -> np.ndarray:
        """
        :param x: numpy array of pwm values in range [0, 4096]
        :return: numpy array of values in range [0, 1]
        """
        return x / 4096

    def _norm2pwm(self, x: np.ndarray) -> np.ndarray:
        """
        :param x: numpy array of values in range [0, 1]
        :return: numpy array of pwm values in range [0, 4096]
        """
        return x * 4096

    def read_position(self) -> np.ndarray:
        """
        :return: numpy array of current joint positions in range [0, 4096]
        """
        return self.d.qpos[:6] # 5-> 6

    def read_velocity(self):
        """
        Reads the joint velocities of the robot.
        :return: list of joint velocities,
        """
        return self.d.qvel

    def read_ee_pos(self, joint_name='end_effector'):
        """
        :param joint_name: name of the end effector joint
        :return: numpy array of end effector position
        """
        joint_id = self.m.body(joint_name).id
        return self.d.geom_xpos[joint_id]

    def set_target_pos(self, target_pos):
        self.d.ctrl = target_pos

    def inverse_kinematics(self, ee_target_pos, rate=0.2, joint_name='end_effector'):
        """
        :param ee_target_pos: numpy array of target end effector position
        :param joint_name: name of the end effector joint
        """
        joint_id = self.m.body(joint_name).id

        # get the current end effector position
        ee_pos = self.d.geom_xpos[joint_id]
        
        # compute the jacobian
        jac = np.zeros((3, self.m.nv))
        mujoco.mj_jacBodyCom(self.m, self.d, jac, None, joint_id)
        
        # compute target joint velocities
        qdot = np.dot(np.linalg.pinv(jac[:, :6]), ee_target_pos - ee_pos) # 5->6 due to increased njoints
        
        # apply the joint velocities
        qpos = self.read_position()
        q_target_pos = qpos + qdot * rate
        return q_target_pos

    def inverse_kinematics_rot(self, ee_target_pos, rate=0.2, joint_name='end_effector'):
        """
        :param ee_target_pos: numpy array of target end effector position
        :param joint_name: name of the end effector joint
        """
        joint_id = self.m.body(joint_name).id

        # 현재 말단 조작기의 위치 및 회전 행렬 얻기
        ee_pos = self.d.geom_xpos[joint_id]
        ee_rot = self.d.geom_xmat[joint_id].reshape(3, 3)  # 3x3 회전 행렬

        # 목표 회전 행렬 정의 (그리퍼가 지면과 수평을 유지하도록)
        # 여기서는 z축이 월드 좌표계의 z축과 일치하도록 설정
        z_axis = np.array([0, 0, 1], dtype=np.float64)  # 월드 좌표계의 z축
        x_axis = np.array([1, 0, 0], dtype=np.float64)  # 임의의 x축 방향 (조정 가능)

        # y축 계산 (정규화 및 직교화 필요)
        y_axis = np.cross(z_axis, x_axis)
        y_axis /= np.linalg.norm(y_axis)
        x_axis = np.cross(y_axis, z_axis)
        x_axis /= np.linalg.norm(x_axis)

        # 목표 회전 행렬 구성
        ee_target_rot = np.column_stack((x_axis, y_axis, z_axis))

        # Jacobian 계산
        jacp = np.zeros((3, self.m.nv))
        jacr = np.zeros((3, self.m.nv))
        mujoco.mj_jacBody(self.m, self.d, jacp, jacr, joint_id)

        # 위치 및 회전 오차 계산
        error_pos = ee_target_pos - ee_pos

        # 회전 행렬을 사원수로 변환
        current_quat = np.zeros(4)
        target_quat = np.zeros(4)
        mujoco.mju_mat2Quat(current_quat, ee_rot.flatten())
        mujoco.mju_mat2Quat(target_quat, ee_target_rot.flatten())

        # 회전 오차 계산 (사원수 곱)
        error_quat = np.zeros(4)
        mujoco.mju_subQuat(error_quat, target_quat, current_quat)

        # 사원수 오차를 각속도로 변환
        error_rot = np.zeros(3)
        mujoco.mju_quat2Vel(error_rot, error_quat, 1.0)

        # 위치 및 회전 오차 결합
        error = np.hstack((error_pos, error_rot))

        # 전체 Jacobian 결합
        jac = np.vstack((jacp, jacr))

        # 관절 속도 계산 (댐핑된 역행렬 사용)
        lambda_identity = 1e-4 * np.eye(6)
        dq = jac.T @ np.linalg.solve(jac @ jac.T + lambda_identity, error)

        # 관절 위치 업데이트
        q = self.d.qpos.copy()
        mujoco.mj_integratePos(self.m, q, dq * rate)

        # 관절 범위 내로 클리핑
        np.clip(q[:6], self.m.jnt_range[:6, 0], self.m.jnt_range[:6, 1], out=q[:6])

        # 제어 신호 설정
        self.d.ctrl[:6] = q[:6]

        # 시뮬레이션 스텝 진행
        mujoco.mj_step(self.m, self.d)

    def inverse_kinematics_rot_backup_6DOF(self, ee_target_pos, ee_target_rot, rate=0.2, joint_name='end_effector'):
        """
        :param ee_target_pos: numpy array of target end effector position
        :param joint_name: name of the end effector joint
        """
        joint_id = self.m.body(joint_name).id

        # get the current end effector position
        ee_pos = self.d.geom_xpos[joint_id]
        ee_rot = self.d.geom_xmat[joint_id]
        error = np.zeros(6)
        error_pos = error[:3]
        error_rot = error[3:]
        site_quat = np.zeros(4)
        site_target_quat = np.zeros(4)
        site_quat_conj = np.zeros(4)
        error_quat = np.zeros(4)

        diag = 1e-4 * np.identity(6)
        integration_dt = 1.0

        # compute the jacobian
        jacp = np.zeros((3, self.m.nv))
        jacr = np.zeros((3, self.m.nv))
        mujoco.mj_jacBodyCom(self.m, self.d, jacp, jacr, joint_id)
        
        # compute target joint velocities
        jac = np.vstack([jacp, jacr])

        # Orientation error.
        mujoco.mju_mat2Quat(site_quat, ee_rot)
        mujoco.mju_mat2Quat(site_target_quat, ee_target_rot)

        mujoco.mju_negQuat(site_quat_conj, site_quat)

        mujoco.mju_mulQuat(error_quat, site_target_quat, site_quat_conj)

        mujoco.mju_quat2Vel(error_rot, error_quat, 1.0)

        error_pos = ee_target_pos - ee_pos
        error = np.hstack([error_pos, error_rot])
        
        dq = jac.T @ np.linalg.solve(jac @ jac.T + diag, error)

        q = self.d.qpos.copy()
        mujoco.mj_integratePos(self.m, q, dq, integration_dt)

        # Set the control signal.
        np.clip(q[:6], *self.m.jnt_range.T[:, :6], out=q[:6])
        self.d.ctrl[:6] = q[:6]

        print(colored(f"Target joint position: {np.round(q[:6], 2)}", 'red'))
        # Step the simulation.
        mujoco.mj_step(self.m, self.d)
        
        return self.d.ctrl[:6]
    

    def inverse_kinematics_rot_backup_5DOF(self, ee_target_pos, ee_target_rot, rate=0.2, joint_name='joint5'):
        """
        :param ee_target_pos: numpy array of target end effector position
        :param joint_name: name of the end effector joint
        """
        joint_id = self.m.body(joint_name).id

        # get the current end effector position
        ee_pos = self.d.geom_xpos[joint_id]
        ee_rot = self.d.geom_xmat[joint_id]
        error = np.zeros(6)
        error_pos = error[:3]
        error_rot = error[3:]
        site_quat = np.zeros(4)
        site_target_quat = np.zeros(4)
        site_quat_conj = np.zeros(4)
        error_quat = np.zeros(4)

        diag = 1e-4 * np.identity(5)
        integration_dt = 1.0

        # compute the jacobian
        jacp = np.zeros((3, self.m.nv))
        jacr = np.zeros((3, self.m.nv))
        mujoco.mj_jacBodyCom(self.m, self.d, jacp, jacr, joint_id)
        
        # compute target joint velocities
        jac = np.vstack([jacp, jacr])

        # Orientation error.
        mujoco.mju_mat2Quat(site_quat, ee_rot)
        mujoco.mju_mat2Quat(site_target_quat, ee_target_rot)

        mujoco.mju_negQuat(site_quat_conj, site_quat)

        mujoco.mju_mulQuat(error_quat, site_target_quat, site_quat_conj)

        mujoco.mju_quat2Vel(error_rot, error_quat, 1.0)

        error_pos = ee_target_pos - ee_pos
        error = np.hstack([error_pos, error_rot])
        
        dq = jac[:5, :5].T @ np.linalg.solve(jac[:5, :5] @ jac[:5, :5].T + diag, error[:5])

        q = self.d.qpos.copy()
        mujoco.mj_integratePos(self.m, q, dq, integration_dt)

        # Set the control signal.
        np.clip(q[:5], *self.m.jnt_range.T[:, :5], out=q[:5])
        self.d.ctrl[:5] = q[:5]

        # Step the simulation.
        '''
        IMPORTANT!!!!: mujoco.mj_step() must be disabled for the real robot
        '''
        mujoco.mj_step(self.m, self.d)
        
        return self.d.ctrl[:5]
    

    

    def inverse_kinematics_rot_backup_4DOF(self, ee_target_pos, ee_target_rot, rate=0.2, joint_name='end_effector'):
        """
        :param ee_target_pos: numpy array of target end effector position
        :param joint_name: name of the end effector joint
        """
        joint_id = self.m.body(joint_name).id

        # get the current end effector position
        ee_pos = self.d.geom_xpos[joint_id]
        ee_rot = self.d.geom_xmat[joint_id]
        error = np.zeros(6)
        error_pos = error[:3]
        error_rot = error[3:]
        site_quat = np.zeros(4)
        site_target_quat = np.zeros(4)
        site_quat_conj = np.zeros(4)
        error_quat = np.zeros(4)

        diag = 1e-4 * np.identity(4)
        integration_dt = 1.0

        # compute the jacobian
        jacp = np.zeros((3, self.m.nv))
        jacr = np.zeros((3, self.m.nv))
        mujoco.mj_jacBodyCom(self.m, self.d, jacp, jacr, joint_id)
        
        # compute target joint velocities
        jac = np.vstack([jacp, jacr])

        # Orientation error.
        mujoco.mju_mat2Quat(site_quat, ee_rot)
        mujoco.mju_mat2Quat(site_target_quat, ee_target_rot)

        mujoco.mju_negQuat(site_quat_conj, site_quat)

        mujoco.mju_mulQuat(error_quat, site_target_quat, site_quat_conj)

        mujoco.mju_quat2Vel(error_rot, error_quat, 1.0)

        error_pos = ee_target_pos - ee_pos
        error = np.hstack([error_pos, error_rot])
        
        dq = jac[:4, :4].T @ np.linalg.solve(jac[:4, :4] @ jac[:4, :4].T + diag, error[:4])

        q = self.d.qpos.copy()
        mujoco.mj_integratePos(self.m, q, dq, integration_dt)

        # Set the control signal.
        np.clip(q[:4], *self.m.jnt_range.T[:, :4], out=q[:4])
        self.d.ctrl[:4] = q[:4]

        print(colored(f"Target joint position: {np.round(q[:4], 2)}", 'red'))
        # Step the simulation.
        mujoco.mj_step(self.m, self.d)
        
        return self.d.ctrl[:4]
    
