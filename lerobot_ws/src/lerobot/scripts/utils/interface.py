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

    # def read_ee_pos(self, joint_name='end_effector'):
    #     """
    #     :param joint_name: name of the end effector joint
    #     :return: numpy array of end effector position
    #     """
    #     joint_id = self.m.body(joint_name).id
    #     return self.d.geom_xpos[joint_id]

    def read_ee_pos(self, joint_name='end_effector'):
        """
        :param joint_name: name of the end effector joint
        :return: numpy array of end effector position
        """
        # 조인트 ID 얻기
        joint_id = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
        
        # 조인트가 속한 바디 ID 얻기
        body_id = self.m.jnt_bodyid[joint_id]
        
        # 해당 바디의 월드 좌표계 위치 반환
        return self.d.xpos[body_id].copy()
    
    def set_target_pos(self, target_pos):
        self.d.ctrl = target_pos

    def inverse_kinematics_5dof_pen_vertical(
        self,
        ee_target_pos: np.ndarray,
        body_name: str = 'link5',
        rate: float = 0.1,
        damping: float = 1e-4
    ):
        """
        5DOF 매니퓰레이터에서
        (1) End-Effector 위치(3자유도) 
        + (2) 펜축(로컬 z축) 이 월드 z축에 수직(= -z 방향) 되도록(회전 2자유도)
        => 총 5차원 부분 IK.

        - 펜을 "아래 방향"으로 수직 정렬한다고 가정.
        (책상 위에서 그림 그린다면 펜이 아래 향하도록)

        :param ee_target_pos:  (3,) 타겟 위치 (x,y,z)
        :param body_name:      End-Effector로 사용하는 body 이름 (link5)
        :param rate:           1스텝당 관절 업데이트 스케일 (커질수록 빠르게 목표로 접근)
        :param damping:        댐핑 계수(역행렬 수치안정)
        """
        # ---------------------------
        # 1) 현재 End-Effector (link5) 위치·자세
        # ---------------------------
        body_id = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_BODY, body_name)

        # (a) 현재 EE 월드 위치 (3,)
        current_pos = self.d.xpos[body_id].copy()

        # (b) 현재 EE 월드 회전행렬 (3x3)
        current_rot = self.d.xmat[body_id].reshape(3,3).copy()

        # ---------------------------
        # 2) 위치 오차 (3차원)
        # ---------------------------
        error_pos = ee_target_pos - current_pos  # shape: (3,)

        # ---------------------------
        # 3) "펜 로컬 z축"이 월드 -z와 평행
        # ---------------------------
        #  - 로컬 z축 = current_rot[:,2]
        #  - 우리는 "아래"(-z)로 향하게 만듬 (책상 위에서 펜끝이 바닥으로)
        pen_axis = current_rot[:, 2].copy()
        desired_axis = np.array([0.0, 0.0, 1.0], dtype=float)

        # 두 벡터가 평행이면 cross=0.
        # 하지만 roll은 자유 -> cross 전체(3D) 중 z축은 무시, x,y만 쓰기
        axis_error_3d = np.cross(pen_axis, desired_axis)  # shape (3,)
        error_rot = axis_error_3d[:2]  # shape (2,)

        # ---------------------------
        # 4) 최종 오차 벡터 = (pos 3) + (orient 2) = 5차원
        # ---------------------------
        error_5d = np.hstack([error_pos, error_rot])  # shape (5,)

        # ---------------------------
        # 5) Jacobian 계산 (position + rotation)
        # ---------------------------
        jacp = np.zeros((3, self.m.nv))  # (3,5)
        jacr = np.zeros((3, self.m.nv))  # (3,5)
        mujoco.mj_jacBodyCom(self.m, self.d, jacp, jacr, body_id)

        # position 3행 + rotation 2행 => (5,5)
        jac_partial = np.vstack([jacp, jacr[:2, :]])  # shape: (5,5)

        # ---------------------------
        # 6) Damped pseudo-inverse로 dq 계산
        # ---------------------------
        lamI = damping * np.eye(5)   # (5,5)
        A = jac_partial @ jac_partial.T + lamI  # (5,5)
        dq = jac_partial.T @ np.linalg.solve(A, error_5d)  # shape (5,)

        # ---------------------------
        # 7) 관절각(qpos) 업데이트
        # ---------------------------
        q = self.d.qpos.copy()
        mujoco.mj_integratePos(self.m, q, dq, rate)  # 1스텝 관절 변화

        # 관절 범위(첫 5개)로 클리핑
        np.clip(q[:5], self.m.jnt_range[:5,0], self.m.jnt_range[:5,1], out=q[:5])

        # ---------------------------
        # 8) 제어값으로 설정 & 시뮬레이션 1스텝
        # ---------------------------
        self.d.ctrl[:5] = q[:5]
        mujoco.mj_step(self.m, self.d)

        return q[:5].copy()
    