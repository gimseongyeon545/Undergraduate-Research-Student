#!/usr/bin/env python3
# 파일명: vr_ik_follower_sub.py
import math, numpy as np, time
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import JointState, Joy
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from moveit_msgs.srv import GetPositionIK
import argparse
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy
from functools import partial
from rclpy.action import ActionClient
from control_msgs.action import GripperCommand
from moveit_msgs.msg import Constraints, OrientationConstraint, PositionConstraint
from shape_msgs.msg import SolidPrimitive
from sensor_msgs.msg import JointState
from std_msgs.msg import Bool


# ---------- SE(3) 유틸 ----------
def quat_to_mat(qx, qy, qz, qw):
    x, y, z, w = qx, qy, qz, qw
    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z
    R = np.array([
        [1 - 2 * (yy + zz), 2 * (xy - wz), 2 * (xz + wy)],
        [2 * (xy + wz), 1 - 2 * (xx + zz), 2 * (yz - wx)],
        [2 * (xz - wy), 2 * (yz + wx), 1 - 2 * (xx + yy)]
    ], dtype=float)
    return R


def mat_to_quat(R):
    m00, m01, m02 = R[0, 0], R[0, 1], R[0, 2]
    m10, m11, m12 = R[1, 0], R[1, 1], R[1, 2]
    m20, m21, m22 = R[2, 0], R[2, 1], R[2, 2]
    tr = m00 + m11 + m22
    tr = max(-1.0, min(3.0, tr))  # ← 추가
    if tr > 0.0:
        S = math.sqrt(max(0.0, tr + 1.0)) * 2  # ← 추가: 음수 루트 방지
        qw = 0.25 * S
        qx = (m21 - m12) / S
        qy = (m02 - m20) / S
        qz = (m10 - m01) / S
    elif (m00 > m11) and (m00 > m22):
        S = math.sqrt(max(0.0, 1.0 + m00 - m11 - m22)) * 2
        qw = (m21 - m12) / S
        qx = 0.25 * S
        qy = (m01 + m10) / S
        qz = (m02 + m20) / S
    elif m11 > m22:
        S = math.sqrt(max(0.0, 1.0 + m11 - m00 - m22)) * 2
        qw = (m02 - m20) / S
        qx = (m01 + m10) / S
        qy = 0.25 * S
        qz = (m12 + m21) / S
    else:
        S = math.sqrt(max(0.0, 1.0 + m22 - m00 - m11)) * 2
        qw = (m10 - m01) / S
        qx = (m02 + m20) / S
        qy = (m12 + m21) / S
        qz = 0.25 * S
    q = quat_normalize([qx, qy, qz, qw])
    return q[0], q[1], q[2], q[3]


def pose_to_T(p: PoseStamped):
    R = quat_to_mat(p.pose.orientation.x, p.pose.orientation.y,
                    p.pose.orientation.z, p.pose.orientation.w)
    t = np.array([p.pose.position.x, p.pose.position.y, p.pose.position.z], dtype=float)
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t
    return T


def T_to_pose(T, frame_id):
    R = T[:3, :3]
    t = T[:3, 3]
    qx, qy, qz, qw = mat_to_quat(R)
    qx, qy, qz, qw = quat_normalize([qx, qy, qz, qw])
    out = PoseStamped()
    out.header.frame_id = frame_id
    out.pose.position.x = float(t[0])
    out.pose.position.y = float(t[1])
    out.pose.position.z = float(t[2])
    out.pose.orientation.x = float(qx)
    out.pose.orientation.y = float(qy)
    out.pose.orientation.z = float(qz)
    out.pose.orientation.w = float(qw)
    return out


def T_inv(T):
    R = T[:3, :3]
    t = T[:3, 3]
    Ti = np.eye(4)
    Ti[:3, :3] = R.T
    Ti[:3, 3] = -R.T @ t
    return Ti


def quat_normalize(q):
    q = np.array(q, dtype=float)
    n = np.linalg.norm(q)
    if n < 1e-12: return np.array([0, 0, 0, 1], dtype=float)
    return q / n


def quat_slerp(q0, q1, alpha):
    q0 = quat_normalize(q0)
    q1 = quat_normalize(q1)
    dot = float(np.dot(q0, q1))
    if dot < 0.0:
        q1 = -q1
        dot = -dot
    if dot > 0.9995:
        q = q0 + alpha * (q1 - q0)
        return quat_normalize(q)
    theta0 = math.acos(dot)
    theta = theta0 * alpha
    sin_t0 = math.sin(theta0)
    s0 = math.sin(theta0 - theta) / sin_t0
    s1 = math.sin(theta) / sin_t0
    return quat_normalize(s0 * q0 + s1 * q1)


def rot_to_axis_angle(R):
    tr = np.clip(np.trace(R), -1.0, 3.0)
    angle = math.acos(np.clip((tr - 1.0) / 2.0, -1.0, 1.0))
    if angle < 1e-9:
        return np.array([1.0, 0, 0]), 0.0
    rx = R[2, 1] - R[1, 2]
    ry = R[0, 2] - R[2, 0]
    rz = R[1, 0] - R[0, 1]
    axis = np.array([rx, ry, rz], dtype=float)
    axis = axis / (2.0 * math.sin(angle) + 1e-12)
    n = np.linalg.norm(axis)
    if n < 1e-12:
        axis = np.array([1.0, 0, 0])
    else:
        axis = axis / n
    return axis, angle


def axis_angle_to_rot(axis, angle):
    ax = np.array(axis, dtype=float)
    ax = ax / (np.linalg.norm(ax) + 1e-12)
    x, y, z = ax
    c = math.cos(angle)
    s = math.sin(angle)
    C = 1 - c
    return np.array([
        [x * x * C + c, x * y * C - z * s, x * z * C + y * s],
        [y * x * C + z * s, y * y * C + c, y * z * C - x * s],
        [z * x * C - y * s, z * y * C + x * s, z * z * C + c]
    ], dtype=float)


def ensure_quat_continuity(q_prev, q_cur):
    """
    q_prev, q_cur: [x,y,z,w]
    프레임 간 연속성 보장(내적<0이면 부호 반전)
    """
    if q_prev is None:
        return np.array(q_cur, dtype=float)
    q_prev = np.array(q_prev, dtype=float)
    q_cur = np.array(q_cur, dtype=float)
    if float(np.dot(q_prev, q_cur)) < 0.0:
        q_cur = -q_cur
    return q_cur


# ---------- 노드 ----------
class VRIKFollower2(Node):
    def __init__(self, group_name, ee_link, controller_topic, base_frame,
                 controller_joint_order, anchor_topic, vr_topic,
                 R_vr2base=None,
                 ik_timeout_sec=0.3, stream_move_time=0.1, max_rate_hz=30.0,
                 joy_topic='/steamvr/right/joy',
                 gripper_controller='/gripper_controller/joint_trajectory',
                 gripper_joint='gripper', grip_min=0.0, grip_max=0.8, grip_step=0.02,
                 gripper_action='/gen3_lite_2f_gripper_controller/gripper_cmd',
                 pos_only=False,
                 j6_name='joint_6', j6_min=-2.0, j6_max=2.0,
                 yaw_gain=2.5, yaw_step_deg=60.0, dq_max=0.08
                 ):

        super().__init__('vr_ik_follower_sub_2')
        self.group_name = group_name
        self.ee_link = ee_link
        self.controller_topic = controller_topic
        self.base_frame = base_frame
        self.controller_joint_order = controller_joint_order
        self.ik_timeout_sec = ik_timeout_sec
        self.stream_move_time = stream_move_time
        self.max_period = 1.0 / max_rate_hz
        self.gripper_action = gripper_action
        self.grip_ac = ActionClient(self, GripperCommand, self.gripper_action)
        self.grip_step = grip_step  # CLI에서 --grip_step 0.01 추천
        self.grip_repeat_dt = 0.07  # 길게 누르면 0.07s마다 한 스텝
        self.grip_repeat_delay = 0.35  # 길게 누를 때 반복 시작 전 딜레이
        self._grip_repeat_timer = None
        self._grip_hold_dir = 0  # -1 닫기, +1 열기, 0 없음
        self.pos_only = bool(pos_only)
        self._seed_js = None

        self.j6_name = j6_name
        self.j6_min = float(j6_min)
        self.j6_max = float(j6_max)
        self.yaw_gain = float(yaw_gain)
        self.yaw_step_cap = math.radians(float(yaw_step_deg))
        self.user_dq_max = float(dq_max)

        self.active_pub = self.create_publisher(Bool, '/vr_active', 10)
        self._active_hysteresis_deadline = 0.0
        self.active_hold_sec = 0.2  # 약간의 여유시간

        # VR→Base 축 정렬 행렬
        self.R_vr2base = np.eye(3) if (R_vr2base is None) else np.array(R_vr2base, dtype=float).reshape(3, 3)
        U, _, Vt = np.linalg.svd(self.R_vr2base)
        self.R_vr2base = U @ Vt
        if np.linalg.det(self.R_vr2base) < 0:
            self.R_vr2base[:, 2] *= -1  # 반사 보정

        self.R_dyn = None
        self.joint_state = None
        self.ee_anchor_T = None
        self.vr_anchor_T = None
        self.last_publish_time = 0.0

        self.alpha_pos = 0.2  # 0.1~0.3 권장 (작을수록 부드러움)
        self.alpha_rot = 0.15  # 0.1~0.25 권장
        self._vr_pos_ema = None
        self._vr_quat_ema = None  # (x,y,z,w)
        self.move_mode = 'none'

        self.grip_ramp_dt = 0.05  # 20Hz
        self._grip_timer = None
        self._grip_goal = None

        self.JOY_TIMEOUT = 2.0  # 0.4 → 2.0s

        self._last_hold_pub = 0.0
        self._ik_busy_deadline = 0.0

        self.js_sub = self.create_subscription(JointState, '/joint_states', self._js_cb, 100)
        self.ik_cli = self.create_client(GetPositionIK, '/compute_ik')

        traj_qos = QoSProfile(
            depth=1, reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE, history=HistoryPolicy.KEEP_LAST
        )
        self.traj_pub = self.create_publisher(JointTrajectory, controller_topic, traj_qos)
        self.ee_pose_cmd_pub = self.create_publisher(PoseStamped, '/ee_pose_cmd', traj_qos)

        self.ik_busy = False

        # ---- Joy / 그리퍼 ----
        self.enable_move = False
        self.joy_topic = joy_topic
        self.create_subscription(Joy, self.joy_topic, self._joy_cb, 10)

        self.gripper_joint = gripper_joint
        self.grip_min = grip_min
        self.grip_max = grip_max
        self.grip_step = grip_step
        self.grip_pos = (grip_min + grip_max) * 0.5
        self.gripper_traj_pub = self.create_publisher(JointTrajectory, gripper_controller, 10)

        self._prev_enable = False
        self._last_vr_T = None
        self._last_target_T = None
        self.cmd_epoch = 0

        latched_qos = QoSProfile(
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST
        )
        self.anchor_echo_pub = self.create_publisher(PoseStamped, '/ee_anchor_echo', latched_qos)

        sensor_qos = QoSProfile(
            depth=10, reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE, history=HistoryPolicy.KEEP_LAST
        )
        self.anchor_sub = self.create_subscription(PoseStamped, anchor_topic, self._anchor_cb, latched_qos)
        self.vr_sub = self.create_subscription(PoseStamped, vr_topic, self._vr_cb, sensor_qos)

        self.get_logger().info('Waiting for /compute_ik ...')
        self.ik_cli.wait_for_service()
        self.get_logger().info('VRIKFollower ready.')
        self.last_joy_time = 0.0

        # 버튼 인덱스
        self.btn_idx = {
            'MENU': 1, 'GRIP': 2, 'TRIG': 3,
            'PAD_CLICK': 4,
            'PAD_TOUCH': 5
        }
        self.last_joy_time = 0.0

        # ★ 쿼터니언 연속성 캐시
        self._q_prev = None

    def _joy_cb(self, msg: Joy):
        axes = list(msg.axes) if msg.axes else []
        buttons = list(msg.buttons) if msg.buttons else []
        if len(axes) < 3: axes += [0.0] * (3 - len(axes))
        if len(buttons) < 6: buttons += [0] * (6 - len(buttons))

        if not hasattr(self, "btn_idx"):
            self.btn_idx = {
                'MENU': 1, 'GRIP': 2, 'TRIG': 3,
                'PAD_CLICK': 4,
                'PAD_TOUCH': 5
            }

        def B(name):
            i = self.btn_idx[name]
            return int(buttons[i]) if i < len(buttons) else 0

        pad_touch = B('PAD_TOUCH')
        pad_click = B('PAD_CLICK')
        menu_btn = B('MENU')
        grip_btn = B('GRIP')
        trig_click = B('TRIG')
        trig_axis = float(axes[2])

        now = self.get_clock().now().nanoseconds * 1e-9

        # 클릭 디바운스
        DEBOUNCE = 0.03
        raw = pad_click

        if not hasattr(self, "_click_raw_prev"):
            self._click_raw_prev = raw
            self._click_stable_since = now
            self._click_state = raw
        else:
            if raw != self._click_raw_prev:
                self._click_raw_prev = raw
                self._click_stable_since = now
            if (now - self._click_stable_since) >= DEBOUNCE and raw != self._click_state:
                self._click_state = raw

        click_on = (self._click_state == 1)
        trig_on = (trig_click == 1) or (trig_axis > 0.85)

        # 모드 결정
        pad_pressed = click_on
        new_mode = 'both' if (pad_pressed and trig_on) else ('pos' if pad_pressed else ('rot' if trig_on else 'none'))

        if self.pos_only:
            if trig_on and not pad_pressed:
                new_mode = 'rot'
            elif pad_pressed and not trig_on:
                new_mode = 'pos'
            else:
                new_mode = 'none'

        if new_mode != getattr(self, "move_mode", 'none'):
            self.get_logger().info(
                f"[joy] mode -> {new_mode} (menu={menu_btn} grip={grip_btn} trigClick={trig_click} "
                f"trigAxis={trig_axis:.2f} touch={pad_touch} click_raw={pad_click} click={int(click_on)})"
            )

        self.move_mode = new_mode
        self.enable_move = (self.move_mode != 'none')
        self.last_joy_time = now

        # 그리퍼 스텝
        if not hasattr(self, "_grip_prev"): self._grip_prev = 0
        if not hasattr(self, "_open_prev"): self._open_prev = 0
        if grip_btn == 1 and self._grip_prev == 0:
            self._grip_step_send(-self.grip_step)
            self._grip_start_repeat(-1)
        elif grip_btn == 0 and self._grip_prev == 1:
            self._grip_stop_repeat()
        self._grip_prev = grip_btn

        if menu_btn == 1 and self._open_prev == 0:
            self._grip_step_send(+self.grip_step)
            self._grip_start_repeat(+1)
        elif menu_btn == 0 and self._open_prev == 1:
            self._grip_stop_repeat()
        self._open_prev = menu_btn

    def _map_R(self, Rvr):
        return self.R_vr2base @ Rvr @ self.R_vr2base.T

    def _map_v(self, v):
        # 위치(delta)는 베이스 축으로만 변환. R_dyn 쓰지 않음
        return self.R_vr2base @ v

    def _send_gripper_action(self, pos: float, effort: float = 30.0):
        if not getattr(self, "_grip_ready", False):
            if not self.grip_ac.wait_for_server(timeout_sec=0.5):
                if not getattr(self, "_grip_warned", False):
                    self.get_logger().error(f'gripper action server not available: {self.gripper_action}')
                    self._grip_warned = True
                return
            self._grip_ready = True
        goal = GripperCommand.Goal()
        goal.command.position = float(pos)
        goal.command.max_effort = float(effort)
        self.grip_ac.send_goal_async(goal)

    def _js_cb(self, msg: JointState):
        self.joint_state = msg

    def _anchor_cb(self, msg: PoseStamped):
        if self.enable_move:
            self.get_logger().warn('Anchor ignored while moving.')
            return
        self.ee_anchor_T = pose_to_T(msg)
        self.vr_anchor_T = None
        self.anchor_echo_pub.publish(msg)

    def _vr_cb(self, msg: PoseStamped):
        # ---- 준비/가드 ----
        if self.ee_anchor_T is None or self.joint_state is None:
            return

        if not hasattr(self, "JOY_TIMEOUT"):      self.JOY_TIMEOUT = 2.0
        if not hasattr(self, "_last_hold_pub"):   self._last_hold_pub = 0.0
        if not hasattr(self, "_prev_enable"):     self._prev_enable = False
        if not hasattr(self, "_vr_pos_ema"):      self._vr_pos_ema = None
        if not hasattr(self, "_vr_quat_ema"):     self._vr_quat_ema = None
        if not hasattr(self, "_last_target_T"):   self._last_target_T = None

        now = self.get_clock().now().nanoseconds * 1e-9

        # IK busy 워치독
        if self.ik_busy and getattr(self, "_ik_busy_deadline", 0.0) and now > self._ik_busy_deadline:
            self.get_logger().warn("IK busy watchdog tripped → reset")
            self.ik_busy = False

        # 현재 VR pose
        T_vr_t = pose_to_T(msg)

        # Joy 타임아웃
        if (now - getattr(self, "last_joy_time", 0.0)) > self.JOY_TIMEOUT and self.enable_move:
            self.enable_move = False
            self.move_mode = 'none'
            if self._prev_enable:
                if self._last_target_T is not None:
                    self.ee_anchor_T = self._last_target_T.copy()
                self.vr_anchor_T = T_vr_t.copy()
                self.ik_busy = False
                if (now - self._last_hold_pub) > 0.25:
                    self._publish_hold_trajectory(0.08)
                    self._last_hold_pub = now

            self._active_hysteresis_deadline = now + self.active_hold_sec
            self.active_pub.publish(Bool(data=True))

            self._prev_enable = False
            return

        # 엣지 처리
        was_enabled = self._prev_enable
        is_rising = self.enable_move and not was_enabled
        is_falling = (not self.enable_move) and was_enabled

        if is_rising:
            if self._last_target_T is not None:
                self.ee_anchor_T = self._last_target_T.copy()
            self.vr_anchor_T = T_vr_t.copy()
            self._vr_pos_ema = None
            self._vr_quat_ema = None
            self.cmd_epoch += 1

        if is_falling:
            self.ik_busy = False
            if (now - self._last_hold_pub) > 0.25:
                self._publish_hold_trajectory(0.08)
                self._last_hold_pub = now
            if self._last_target_T is not None:
                self.ee_anchor_T = self._last_target_T.copy()
            self.vr_anchor_T = T_vr_t.copy()

        self._prev_enable = self.enable_move

        # 첫 앵커 초기화
        if self.vr_anchor_T is None:
            self.vr_anchor_T = T_vr_t.copy()
            return

        # 주기 제한
        if (now - self.last_publish_time) < self.max_period:
            return
        self.last_publish_time = now

        # IK 과부하 방지
        if self.ik_busy:
            return

        # ---- 원본 VR 쿼터니언 사용 + 연속화 ----
        pos_now = T_vr_t[:3, 3].copy()
        q_raw = np.array([
            msg.pose.orientation.x,
            msg.pose.orientation.y,
            msg.pose.orientation.z,
            msg.pose.orientation.w
        ], dtype=float)
        q_raw = quat_normalize(q_raw)
        q_now = ensure_quat_continuity(self._q_prev, q_raw)
        self._q_prev = q_now.copy()

        # (필요 시 여기서 slerp/EMA 가능)
        self._vr_pos_ema = pos_now
        self._vr_quat_ema = q_now

        R_now = quat_to_mat(*self._vr_quat_ema)
        T_vr_t[:3, :3] = R_now
        T_vr_t[:3, 3] = self._vr_pos_ema

        # ---- VR 기준 대비 델타 (위치: 월드Δ→베이스) ----
        d_vr_world = T_vr_t[:3, 3] - self.vr_anchor_T[:3, 3]
        d_base = self._map_v(d_vr_world)

        trans_deadzone = 0.0005  # 0.5 mm
        trans_step_cap = 0.30    # 10 cm / tick
        move_scale = 0.6

        n = float(np.linalg.norm(d_base))
        if n < trans_deadzone:
            d_base_scaled = np.zeros(3)
        else:
            if n > trans_step_cap:
                d_base = d_base * (trans_step_cap / n)
            d_base_scaled = move_scale * d_base

        # ---- 회전 델타 (도구 Z 성분 기반 yaw 제어) ----
        dT = T_inv(self.vr_anchor_T) @ T_vr_t
        dR_vr = dT[:3, :3]
        dR_base = self._map_R(dR_vr)

        R_anchor = self.ee_anchor_T[:3, :3]
        p_anchor = self.ee_anchor_T[:3, 3]

        axis, ang = rot_to_axis_angle(dR_base)
        rot_deadzone = math.radians(1.0)
        rot_step_cap = math.radians(30.0)
        rot_gain = 1.0

        # 손목(z)과 base z 정렬도
        wrist_align = abs(float(np.dot(R_anchor[:, 2], np.array([0.0, 0.0, 1.0]))))
        if wrist_align > 0.96:
            rot_step_cap = math.radians(30.0)
            rot_gain = 1.0
        else:
            rot_step_cap = math.radians(30.0)
            rot_gain = 1.0

        if ang < rot_deadzone:
            rotvec_base = np.zeros(3)
        else:
            ang = min(ang, rot_step_cap) * rot_gain
            rotvec_base = axis * ang

        # pos_only면 rot 모드 외엔 회전 무시
        if self.pos_only and self.move_mode != 'rot':
            rotvec_base[:] = 0.0

        # 툴 Z(파란축) 성분만 추출 → yaw 제어
        tool_z_in_base = R_anchor[:, 2]
        ang_about_toolZ = float(np.dot(rotvec_base, tool_z_in_base))  # 라디안(+/-)

        # yaw 증폭 & per-tick 상한
        ang_about_toolZ *= self.yaw_gain
        if ang_about_toolZ >= 0.0:
            ang_about_toolZ = min(ang_about_toolZ, self.yaw_step_cap)
        else:
            ang_about_toolZ = max(ang_about_toolZ, -self.yaw_step_cap)

        # joint6 한계 보정
        idx_map = {n: i for i, n in enumerate(self.joint_state.name)}
        if self.j6_name in idx_map:
            q6_cur = float(self.joint_state.position[idx_map[self.j6_name]])
        else:
            q6_cur = None

        if q6_cur is not None:
            MARGIN = 0.01
            if ang_about_toolZ > 0.0:
                allowable = max(0.0, self.j6_max - (q6_cur + MARGIN))
                ang_about_toolZ = min(ang_about_toolZ, allowable)
            elif ang_about_toolZ < 0.0:
                allowable = min(0.0, self.j6_min - (q6_cur - MARGIN))
                ang_about_toolZ = max(ang_about_toolZ, allowable)

        moved_pos = (np.linalg.norm(d_base_scaled) >= trans_deadzone)
        moved_rot = (abs(ang_about_toolZ) >= rot_deadzone) if self.pos_only else (ang >= rot_deadzone)

        is_active = self.enable_move and (
            (self.move_mode == 'pos' and moved_pos) or
            (self.move_mode == 'rot' and moved_rot) or
            (self.move_mode == 'both' and (moved_pos or moved_rot))
        )

        if is_active:
            self._active_hysteresis_deadline = now + self.active_hold_sec

        # 히스테리시스: 막 멈춘 직후에도 잠깐 True 유지
        self.active_pub.publish(Bool(data=(now <= self._active_hysteresis_deadline)))

        if not self.enable_move:
            return

        # 회전 델타 최종 구성 (툴 Z만)
        if abs(ang_about_toolZ) < rot_deadzone:
            dR_tool_z = np.eye(3)
        else:
            dR_tool_z = axis_angle_to_rot([0, 0, 1], ang_about_toolZ)

        # 델타 없으면 조기 종료
        pos_is_zero = (np.linalg.norm(d_base_scaled) < 1e-9)
        rot_is_zero = (abs(ang_about_toolZ) < rot_deadzone)

        if (self.move_mode == 'pos' and pos_is_zero) or \
           (self.move_mode == 'rot' and rot_is_zero) or \
           (self.move_mode == 'both' and pos_is_zero and rot_is_zero):
            return

        if self.move_mode == 'pos':
            R_target = R_anchor
            p_target = p_anchor + d_base_scaled
        elif self.move_mode == 'rot':
            if self.pos_only:
                R_target = R_anchor @ dR_tool_z
            else:
                dR_tool = R_anchor.T @ dR_base @ R_anchor
                R_target = R_anchor @ dR_tool
            p_target = p_anchor
        elif self.move_mode == 'both':
            dR_tool = R_anchor.T @ dR_base @ R_anchor
            R_target = R_anchor @ dR_tool
            p_target = p_anchor + d_base_scaled
        else:
            return

        T_target = np.eye(4)
        T_target[:3, :3] = R_target
        T_target[:3, 3] = p_target

        target_pose = T_to_pose(T_target, self.base_frame)
        target_pose.header.stamp = self.get_clock().now().to_msg()

        local_epoch = self.cmd_epoch
        self._ik_request_async(target_pose, self.stream_move_time, local_epoch)

        self.ee_pose_cmd_pub.publish(target_pose)

        self.get_logger().info(
            f"[vr] mode={self.move_mode} | |d_base|={np.linalg.norm(d_base_scaled):.4f} "
            f"ang={math.degrees(ang):.2f}° align={wrist_align:.3f}"
        )

    def _publish_hold_trajectory(self, hold_time: float = 0.08):
        if self.joint_state is None:
            return

        idx = {n: i for i, n in enumerate(self.joint_state.name)}
        cur = []
        for jn in self.controller_joint_order:
            if jn not in idx:
                self.get_logger().warn(f"[hold] joint {jn} not in joint_state; using 0.0")
                cur.append(0.0)
            else:
                cur.append(float(self.joint_state.position[idx[jn]]))

        traj = JointTrajectory()
        now = self.get_clock().now().to_msg()
        now.nanosec += int(0.05 * 1e9)
        if now.nanosec >= 1_000_000_000:
            now.sec += 1
            now.nanosec -= 1_000_000_000

        traj.header.stamp = now
        traj.joint_names = self.controller_joint_order

        pt = JointTrajectoryPoint()
        pt.positions = cur
        pt.velocities = [0.0] * len(cur)
        pt.accelerations = [0.0] * len(cur)
        pt.time_from_start.sec = int(hold_time)
        pt.time_from_start.nanosec = int((hold_time - int(hold_time)) * 1e9)

        traj.points.append(pt)
        self.traj_pub.publish(traj)
        self.get_logger().info(f"[hold] published hold trajectory ({hold_time:.2f}s)")

    def _ik_and_publish(self, target: PoseStamped, move_time):
        req = GetPositionIK.Request()
        req.ik_request.group_name = self.group_name
        req.ik_request.ik_link_name = self.ee_link
        req.ik_request.robot_state.joint_state = self.joint_state
        req.ik_request.pose_stamped = target
        req.ik_request.timeout.sec = int(self.ik_timeout_sec)
        req.ik_request.timeout.nanosec = int((self.ik_timeout_sec - int(self.ik_timeout_sec)) * 1e9)
        req.ik_request.avoid_collisions = False

        fut = self.ik_cli.call_async(req)
        rclpy.spin_until_future_complete(self, fut)
        if not fut.result():
            self.get_logger().warn('IK call failed')
            return
        res = fut.result()
        if res.error_code.val != 1:
            self.get_logger().warn(f'IK failed, code={res.error_code.val}')
            return

        name_to_pos = dict(zip(res.solution.joint_state.name,
                               res.solution.joint_state.position))
        positions = []
        for jn in self.controller_joint_order:
            if jn not in name_to_pos:
                self.get_logger().error(f'IK solution missing joint: {jn}')
                return
            positions.append(float(name_to_pos[jn]))

        idx = {n: i for i, n in enumerate(self.joint_state.name)}
        cur = [self.joint_state.position[idx[j]] for j in self.controller_joint_order]
        import numpy as np
        delta = float(np.linalg.norm(np.array(positions) - np.array(cur)))
        self.get_logger().info(f'delta(q)={delta:.4f}')

        traj = JointTrajectory()
        now = self.get_clock().now().to_msg()
        now.nanosec += int(0.05 * 1e9)
        if now.nanosec >= 1_000_000_000:
            now.sec += 1
            now.nanosec -= 1_000_000_000
        traj.header.stamp = now
        traj.joint_names = self.controller_joint_order
        pt = JointTrajectoryPoint()
        pt.positions = positions
        pt.time_from_start.sec = int(move_time)
        pt.time_from_start.nanosec = int((move_time - int(move_time)) * 1e9)
        traj.points.append(pt)
        self.traj_pub.publish(traj)
        self.get_logger().info('sent traj')

    def _ik_request_async(self, target: PoseStamped, move_time: float, epoch: int):
        req = GetPositionIK.Request()

        req.ik_request.constraints = Constraints()

        req.ik_request.group_name = self.group_name
        req.ik_request.ik_link_name = self.ee_link
        req.ik_request.robot_state.joint_state = self._seed_js or self.joint_state
        req.ik_request.pose_stamped = target
        req.ik_request.timeout.sec = int(self.ik_timeout_sec)
        req.ik_request.timeout.nanosec = int((self.ik_timeout_sec - int(self.ik_timeout_sec)) * 1e9)
        req.ik_request.avoid_collisions = False

        # Orientation tolerance
        oc = OrientationConstraint()
        oc.header.frame_id = self.base_frame
        oc.link_name = self.ee_link

        qx = target.pose.orientation.x
        qy = target.pose.orientation.y
        qz = target.pose.orientation.z
        qw = target.pose.orientation.w
        qx, qy, qz, qw = quat_normalize([qx, qy, qz, qw])
        oc.orientation.x, oc.orientation.y, oc.orientation.z, oc.orientation.w = qx, qy, qz, qw

        if self.move_mode == 'rot':
            oc.absolute_x_axis_tolerance = math.pi
            oc.absolute_y_axis_tolerance = math.pi
            oc.absolute_z_axis_tolerance = math.radians(6)
        elif self.move_mode == 'pos':
            loose = math.radians(15)
            oc.absolute_x_axis_tolerance = loose
            oc.absolute_y_axis_tolerance = loose
            oc.absolute_z_axis_tolerance = loose
        else:  # both
            mid = math.radians(8)
            oc.absolute_x_axis_tolerance = mid
            oc.absolute_y_axis_tolerance = mid
            oc.absolute_z_axis_tolerance = mid

        oc.weight = 1.0
        req.ik_request.constraints.orientation_constraints = [oc]
        req.ik_request.constraints.position_constraints = []
        req.ik_request.avoid_collisions = True

        self.ik_busy = True
        fut = self.ik_cli.call_async(req)
        fut.add_done_callback(partial(self._on_ik_done,
                                      target=target,
                                      move_time=move_time,
                                      epoch=epoch))
        self._ik_busy_deadline = self.get_clock().now().nanoseconds * 1e-9 + self.ik_timeout_sec + 0.05

    def _grip_step_send(self, delta):
        new = float(np.clip(self.grip_pos + delta, self.grip_min, self.grip_max))
        if abs(new - self.grip_pos) < 1e-6:
            return
        self.grip_pos = new
        self._send_gripper_action(new)
        self.get_logger().info(f"[grip] pos={self.grip_pos:.3f}")

        now = self.get_clock().now().nanoseconds * 1e-9
        self._active_hysteresis_deadline = max(self._active_hysteresis_deadline, now + self.active_hold_sec)
        self.active_pub.publish(Bool(data=True))

    def _grip_repeat_cb(self):
        if self._grip_hold_dir != 0:
            self._grip_step_send(self._grip_hold_dir * self.grip_step)

    def _grip_start_repeat(self, dir_sign):
        self._grip_stop_repeat()
        self._grip_hold_dir = dir_sign

        def _start_periodic():
            if self._grip_repeat_timer:
                self._grip_repeat_timer.cancel()
            self._grip_repeat_timer = self.create_timer(self.grip_repeat_dt, self._grip_repeat_cb)

        self._grip_repeat_timer = self.create_timer(self.grip_repeat_delay, _start_periodic)

        now = self.get_clock().now().nanoseconds * 1e-9
        self._active_hysteresis_deadline = max(self._active_hysteresis_deadline, now + self.active_hold_sec)
        self.active_pub.publish(Bool(data=True))

    def _grip_stop_repeat(self):
        if self._grip_repeat_timer:
            self._grip_repeat_timer.cancel()
            self._grip_repeat_timer = None
        self._grip_hold_dir = 0

    def _on_ik_done(self, future, *, target: PoseStamped, move_time: float, epoch: int):
        if epoch + 1 < self.cmd_epoch:
            try:
                _ = future.result()
            except Exception:
                pass
            self.ik_busy = False
            self.get_logger().info(f'IK result dropped (stale epoch: got {epoch}, cur {self.cmd_epoch})')
            return

        try:
            res = future.result()
        except Exception as e:
            self.get_logger().warn(f'IK call failed: {e}')
            self.ik_busy = False
            return

        if res is None or res.error_code.val != 1:
            self.get_logger().warn(f'IK failed, code={getattr(res, "error_code", None)}')
            self.ik_busy = False
            return

        name_to_pos = dict(zip(res.solution.joint_state.name,
                               res.solution.joint_state.position))
        positions = []
        for jn in self.controller_joint_order:
            if jn not in name_to_pos:
                self.get_logger().error(f'IK solution missing joint: {jn}')
                self.ik_busy = False
                return
            positions.append(float(name_to_pos[jn]))

        idx = {n: i for i, n in enumerate(self.joint_state.name)}
        cur = np.array([self.joint_state.position[idx[j]] for j in self.controller_joint_order])

        if self.pos_only:
            positions = [name_to_pos[j] for j in self.controller_joint_order]
            dq = np.array(positions) - cur
        else:
            dq = np.array([name_to_pos[j] for j in self.controller_joint_order]) - cur
            dq_max = self.user_dq_max
            dq = np.clip(dq, -dq_max, dq_max)
            positions = (cur + dq).tolist()

        self.get_logger().info(f"[ik] ||dq||={float(np.linalg.norm(dq)):.6f}")

        if float(np.linalg.norm(dq)) < 1e-4:
            self.ik_busy = False
            return

        self._last_target_T = pose_to_T(target)

        delta = float(np.linalg.norm(np.array(positions) - cur))
        self.get_logger().info(f'delta(q)={delta:.4f}; q_target={positions}')

        traj = JointTrajectory()
        now = self.get_clock().now().to_msg()
        now.nanosec += int(0.05 * 1e9)
        if now.nanosec >= 1_000_000_000:
            now.sec += 1
            now.nanosec -= 1_000_000_000
        traj.header.stamp = now
        traj.joint_names = self.controller_joint_order
        pt = JointTrajectoryPoint()
        pt.positions = positions
        pt.time_from_start.sec = int(move_time)
        pt.time_from_start.nanosec = int((move_time - int(move_time)) * 1e9)
        traj.points.append(pt)

        seed = JointState()
        seed.name = list(self.controller_joint_order)
        seed.position = list(positions)
        self._seed_js = seed

        self.traj_pub.publish(traj)
        self.get_logger().info('sent traj')
        self.ik_busy = False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--group', default='arm')
    parser.add_argument('--ee', default='end_effector_link')
    parser.add_argument('--controller', default='/joint_trajectory_controller/joint_trajectory')
    parser.add_argument('--base', default='base_link')
    parser.add_argument('--anchor_topic', default='/ee_anchor')
    parser.add_argument('--vr_topic', default='/steamvr/right')
    parser.add_argument('--ik_timeout', type=float, default=1.0)
    parser.add_argument('--rate', type=float, default=30.0)
    parser.add_argument('--move_time', type=float, default=0.1)
    parser.add_argument('--j1', default='joint_1')
    parser.add_argument('--j2', default='joint_2')
    parser.add_argument('--j3', default='joint_3')
    parser.add_argument('--j4', default='joint_4')
    parser.add_argument('--j5', default='joint_5')
    parser.add_argument('--j6', default='joint_6')
    parser.add_argument('--joy_topic', default='/steamvr/right/joy')
    parser.add_argument('--gripper_controller', default='/gripper_controller/joint_trajectory')
    parser.add_argument('--gripper_joint', default='gripper')
    parser.add_argument('--grip_min', type=float, default=0.0)
    parser.add_argument('--grip_max', type=float, default=0.8)
    parser.add_argument('--grip_step', type=float, default=0.02)
    parser.add_argument('--R_vr2base', nargs=9, type=float, default=None)
    parser.add_argument('--gripper_action', default='/gen3_lite_2f_gripper_controller/gripper_cmd')
    parser.add_argument('--pos_only', action='store_true',
                        help='Ignore VR rotations; hold EE orientation fixed (anchor).')
    parser.add_argument('--j6_name', default='joint_6')
    parser.add_argument('--j6_min', type=float, default=-2.0)
    parser.add_argument('--j6_max', type=float, default=2.0)
    parser.add_argument('--yaw_gain', type=float, default=3.5, help='per-tick yaw amplification gain about tool Z')
    parser.add_argument('--yaw_step_deg', type=float, default=60.0, help='per-tick yaw cap in degrees (tool Z only)')
    parser.add_argument('--dq_max', type=float, default=0.2, help='per-joint per-tick clamp (rad)')

    args = parser.parse_args()

    joint_order = [args.j1, args.j2, args.j3, args.j4, args.j5, args.j6]

    rclpy.init()

    node = VRIKFollower2(
        args.group, args.ee, args.controller, args.base,
        joint_order,
        args.anchor_topic, args.vr_topic,
        R_vr2base=args.R_vr2base,
        ik_timeout_sec=args.ik_timeout,
        stream_move_time=args.move_time,
        max_rate_hz=args.rate,
        joy_topic=args.joy_topic,
        gripper_controller=args.gripper_controller,
        gripper_joint=args.gripper_joint,
        grip_min=args.grip_min, grip_max=args.grip_max,
        grip_step=args.grip_step,
        gripper_action=args.gripper_action,
        pos_only=args.pos_only,
        j6_name=args.j6_name,
        j6_min=args.j6_min,
        j6_max=args.j6_max,
        yaw_gain=args.yaw_gain,
        yaw_step_deg=args.yaw_step_deg,
        dq_max=args.dq_max
    )

    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
