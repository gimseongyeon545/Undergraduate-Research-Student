#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, time
from typing import List, Tuple, Optional, Dict
import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy, QoSDurabilityPolicy
from rclpy.action import ActionClient
from rclpy.executors import SingleThreadedExecutor

from std_msgs.msg import Header
from sensor_msgs.msg import Image, JointState
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from control_msgs.action import GripperCommand as GripperCommandAction

import torch
import torch.nn as nn
import cv2
try:
    from cv_bridge import CvBridge
except Exception:
    CvBridge = None

# === 모델 ===
from gen3lite_teleop_wo_servo.resnet_mlp import JointKeyframeRegressor


# ================== DH / FK (너가 준 그대로) ==================
USE_MODIFIED_DH = False
DH_ALPHA = np.array([np.pi/2, np.pi, np.pi/2, np.pi/2, np.pi/2, 0.0], dtype=float)
DH_A     = np.array([0.0,     0.28,  0.0,     0.0,     0.0,     0.0 ], dtype=float)
DH_D     = np.array([0.2433,  0.03,  0.02,    0.245,   0.057,   0.235], dtype=float)
THETA_OFFSETS = np.array([0.0, np.pi/2, np.pi/2, np.pi/2, np.pi, np.pi/2], dtype=float)

def Rx(a):
    c,s = np.cos(a), np.sin(a)
    return np.array([[1,0,0,0],[0,c,-s,0],[0,s,c,0],[0,0,0,1]], float)
def Rz(t):
    c,s = np.cos(t), np.sin(t)
    return np.array([[c,-s,0,0],[s,c,0,0],[0,0,1,0],[0,0,0,1]], float)
def Tx(x): return np.array([[1,0,0,x],[0,1,0,0],[0,0,1,0],[0,0,0,1]], float)
def Tz(z): return np.array([[1,0,0,0],[0,1,0,0],[0,0,1,z],[0,0,0,1]], float)
def A_i(a, alpha, d, theta, modified):
    return (Tx(a) @ Rx(alpha) @ Rz(theta) @ Tz(d)) if modified else (Rz(theta) @ Tz(d) @ Tx(a) @ Rx(alpha))

BASE_T = np.eye(4)
TOOL_T = Tz(-0.13)

def fk_dh_q6(q6_rad: np.ndarray) -> np.ndarray:
    """q6(라디안) -> 4x4 T"""
    T = BASE_T.copy()
    for i in range(6):
        th = float(q6_rad[i]) + float(THETA_OFFSETS[i])
        T = T @ A_i(DH_A[i], DH_ALPHA[i], DH_D[i], th, USE_MODIFIED_DH)
    return T @ TOOL_T

def fk_xyz(q6_rad: np.ndarray) -> np.ndarray:
    T = fk_dh_q6(q6_rad)
    return T[:3, 3]

# ================== 유틸 ==================
def to_numpy_rgb(msg: Image, target_hw: Optional[Tuple[int,int]]) -> np.ndarray:
    if CvBridge is None:
        raise RuntimeError("cv_bridge not available")
    bgr = CvBridge().imgmsg_to_cv2(msg, desired_encoding="bgr8")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    if target_hw is not None:
        H, W = target_hw
        if rgb.shape[:2] != (H, W):
            rgb = cv2.resize(rgb, (W, H), interpolation=cv2.INTER_AREA)
    return rgb.astype(np.uint8)

def denorm_minmax(y: np.ndarray, mn: np.ndarray, mx: np.ndarray) -> np.ndarray:
    y = np.clip(y, -1.0, 1.0)
    return (y + 1.0) * 0.5 * (mx - mn) + mn

def angle_wrap_diff(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    d = a - b
    return (d + np.pi) % (2*np.pi) - np.pi

# ================== 트리 자료구조 ==================
class FKTree:
    """
    nodes: dict[str -> q(6,)]
    parent: dict[str -> str or None]  # root는 None
    precomputed xyz: dict[str -> (3,)]
    """
    def __init__(self, nodes_q: Dict[str, np.ndarray], parent: Dict[str, Optional[str]]):
        self.nodes_q = {k: np.array(v, dtype=np.float32) for k,v in nodes_q.items()}
        self.parent  = parent
        self.nodes_xyz = {k: fk_xyz(self.nodes_q[k]) for k in self.nodes_q.keys()}

    def nearest_by_xyz(self, xyz: np.ndarray) -> str:
        keys = list(self.nodes_xyz.keys())
        d = [np.linalg.norm(self.nodes_xyz[k] - xyz) for k in keys]
        return keys[int(np.argmin(d))]

    def path_to_root(self, start_key: str) -> List[str]:
        path = [start_key]
        cur = start_key
        while self.parent.get(cur, None) is not None:
            cur = self.parent[cur]
            path.append(cur)
        return path  # start -> ... -> root

# ====== 너가 준 노드(q는 라디안 가정; joint 순서는 질문의 joint_name 순서) ======
NODES_Q = {
    "p1":  np.array([-0.2736046482792531, -0.653531209389957,  1.529001741911629,  1.0971962701339633, 1.3695114158133996, -1.8006367162486585], dtype=np.float32),
    "p3":  np.array([ 0.10184940009677237,-0.502113457187976,  1.5133450178630534, 0.9870524548221392, 1.6398632791372294, -1.4142051178593276], dtype=np.float32),
    "p4":  np.array([ 0.32931824459981934,-0.5289895462768373, 1.5061358407922085, 0.9147031564362835, 1.692680687772335, -1.1754378124316194], dtype=np.float32),
    "p5":  np.array([ 0.33437159276824474,-0.47564429911367867,1.4834989715265352, 0.6417033120405878, 2.0192849049052546, -1.1380427696691626], dtype=np.float32),
    "p6":  np.array([ 0.5286867448608953, -0.6588522052479355, 1.4909334520256003, 0.7369483389500727, 1.74977726364698,  -0.9559267595818381], dtype=np.float32),

    "p2":  np.array([ 0.26716832055674916,-0.734631388801704,  1.5134297063857178, 1.1450482149166161, 1.2609724213744573, -1.2515562827082638], dtype=np.float32),
    "p7":  np.array([-0.10184940009677237,-0.502113457187976,  1.5133450178630534, 0.9870524548221392, 1.6398632791372294, -1.4142051178593276], dtype=np.float32),
    "p8":  np.array([-0.32931824459981934,-0.5289895462768373, 1.5061358407922085, 0.9147031564362835, 1.692680687772335, -1.1754378124316194], dtype=np.float32),
    "p9":  np.array([-0.33437159276824474,-0.47564429911367867,1.4834989715265352, 0.6417033120405878, 2.0192849049052546, -1.1380427696691626], dtype=np.float32),
    "p10": np.array([-0.5286867448608953, -0.6588522052479355, 1.4909334520256003, 0.7369483389500727, 1.74977726364698,  -0.9559267595818380], dtype=np.float32),
}

# ====== parent — : p1(root_L), p2(root_R)
PARENT_LEFT = {
    "p1": None,                   # root (cup1)
    "p7": "p1",
    "p3": "p7",
    "p4": "p3",
    "p5": "p3",
    "p6": "p5",
    "p8": "p1",
    "p9": "p8",
    "p10": "p8",
}

PARENT_RIGHT = {
    "p2": None,                   # root (cup2)
    "p4": "p2",
    "p6": "p4",
    "p5": "p4",
    "p3": "p2",
    "p7": "p3",
    "p8": "p7",
    "p9": "p7",
    "p10":"p9",
}

# ================== 메인 노드 ==================
class KeyframeTreeFKRunner(Node):
    """
    1) RGB 한 장 → (q1, q2) 예측
    2) q1 이동 → 집기
    3) q1의 FK(xyz)와 트리 각 노드 FK(xyz) 거리 비교 → 가장 가까운 노드 N
    4) N → parent → … → root 순으로 이동
    5) q2 이동 → 놓기 → 종료
    """
    def __init__(self):
        super().__init__("keyframe_tree_fk_runner")

        # --- params ---
        self.declare_parameter("model_path", "")
        self.declare_parameter("arm_stats_path", "")
        self.declare_parameter("color_topic", "/camera/d435/color/image_raw")
        self.declare_parameter("joint_states_topic", "/joint_states")
        self.declare_parameter("traj_topic", "/joint_trajectory_controller/joint_trajectory")
        self.declare_parameter("gripper_action_name", "/gen3_lite_2f_gripper_controller/gripper_cmd")
        self.declare_parameter("joint_names", ["joint_1","joint_2","joint_4","joint_5","joint_3","joint_6"])
        self.declare_parameter("img_h", 240); self.declare_parameter("img_w", 424)
        self.declare_parameter("move_time", 8.0)
        self.declare_parameter("reach_tol", 0.02)
        self.declare_parameter("reach_stable_count", 3)
        self.declare_parameter("reach_timeout", 6.0)
        self.declare_parameter("grip_open", 0.0); self.declare_parameter("grip_close", 0.8)
        self.declare_parameter("grip_timeout", 3.0)
        self.declare_parameter("device", "cuda")
        self.declare_parameter("fk_y_epsilon", 0.0)  # y≈0일 때의 임계값(0이면 순수 부호)

        # --- load params ---
        self.model_path = self.get_parameter("model_path").value
        self.arm_stats_path = self.get_parameter("arm_stats_path").value
        self.color_topic = self.get_parameter("color_topic").value
        self.joint_states_topic = self.get_parameter("joint_states_topic").value
        self.traj_topic = self.get_parameter("traj_topic").value
        self.gripper_action_name = self.get_parameter("gripper_action_name").value
        self.joint_names = list(self.get_parameter("joint_names").value)
        self.img_h = int(self.get_parameter("img_h").value)
        self.img_w = int(self.get_parameter("img_w").value)
        self.move_time = float(self.get_parameter("move_time").value)
        self.reach_tol = float(self.get_parameter("reach_tol").value)
        self.reach_stable_count = int(self.get_parameter("reach_stable_count").value)
        self.reach_timeout = float(self.get_parameter("reach_timeout").value)
        self.grip_open = float(self.get_parameter("grip_open").value)
        self.grip_close = float(self.get_parameter("grip_close").value)
        self.grip_timeout = float(self.get_parameter("grip_timeout").value)
        dev = self.get_parameter("device").value
        self.device = torch.device(dev if (dev == "cuda" and torch.cuda.is_available()) else "cpu")
        self.fk_y_epsilon = float(self.get_parameter("fk_y_epsilon").value)

        # --- ros io ---
        qos_sensor = QoSProfile(reliability=QoSReliabilityPolicy.BEST_EFFORT,
                                history=QoSHistoryPolicy.KEEP_LAST,
                                depth=10,
                                durability=QoSDurabilityPolicy.VOLATILE)
        self.sub_rgb = self.create_subscription(Image, self.color_topic, self.cb_rgb, qos_sensor)
        self.sub_js  = self.create_subscription(JointState, self.joint_states_topic, self.cb_joint_states, qos_sensor)
        self.pub_traj = self.create_publisher(JointTrajectory, self.traj_topic, 10)
        self.grip_client = ActionClient(self, GripperCommandAction, self.gripper_action_name)

        # --- state ---
        self.rgb_first: Optional[np.ndarray] = None
        self.q_current6 = np.zeros(6, dtype=np.float32)
        self.state = "WAIT_RGB"; self.state_ts = time.time(); self.reach_counter = 0

        # predictions
        self.q1: Optional[np.ndarray] = None
        self.q2: Optional[np.ndarray] = None

        # tree
        self.forest = {
            "left": FKTree(NODES_Q, PARENT_LEFT),
            "right": FKTree(NODES_Q, PARENT_RIGHT),
        }
        self.tree = None  # 선택된 숲이 여기에 들어감
        self.climb_path: List[str] = []
        self.climb_idx = 0

        # model/stats
        self.model = self._load_model(self.model_path)
        self.arm_min, self.arm_max = self._load_arm_stats(self.arm_stats_path)

        self.get_logger().info("[runner] ready")
        self.timer = self.create_timer(0.05, self.tick)

    def _wait_for_traj_connection(self, timeout=5.0):
        import time
        t0 = time.time()
        while time.time() - t0 < timeout:
            if self.pub_traj.get_subscription_count() > 0:
                return True
            time.sleep(0.05)
        self.get_logger().warn("[traj] no subscribers on trajectory topic")
        return False

    # ---- loaders ----
    def _load_model(self, path: str) -> nn.Module:
        if not path or not os.path.exists(path):
            raise FileNotFoundError(f"model_path not found: {path}")
        model = JointKeyframeRegressor(pretrained=False).to(self.device)
        ckpt = torch.load(path, map_location="cpu")
        state_dict = ckpt.get("model_state", ckpt.get("state_dict", ckpt)) if isinstance(ckpt, dict) else ckpt
        def strip(sd,pfx):
            if all(k.startswith(pfx) for k in sd.keys()):
                return {k[len(pfx):]:v for k,v in sd.items()}
            return sd
        state_dict = strip(strip(state_dict, "model."), "module.")
        model.load_state_dict(state_dict, strict=True)
        model.eval()
        return model

    def _load_arm_stats(self, path: str):
        if not path or not os.path.exists(path): return None, None
        z = np.load(path, allow_pickle=False)
        if "arm_min" in z.files and "arm_max" in z.files:
            return z["arm_min"].astype(np.float32), z["arm_max"].astype(np.float32)
        return None, None

    # ---- callbacks ----
    def cb_rgb(self, msg: Image):
        if self.rgb_first is None:
            try:
                self.rgb_first = to_numpy_rgb(msg, (self.img_h, self.img_w))
                self.get_logger().info(f"[rgb] first frame {self.rgb_first.shape}")
            except Exception as e:
                self.get_logger().warn(f"[rgb] parse error: {e}")

    def cb_joint_states(self, msg: JointState):
        name_to_idx = {n: i for i, n in enumerate(msg.name)}
        q6 = np.zeros(6, dtype=np.float32)
        for j, n in enumerate(self.joint_names):
            if n in name_to_idx:
                q6[j] = float(msg.position[name_to_idx[n]])
        self.q_current6 = q6

    # ---- main loop ----
    @torch.no_grad()
    def tick(self):
        now = time.time()

        if self.state == "WAIT_RGB":
            if self.rgb_first is None: return
            self.state = "PREDICT"; self.state_ts = now
            self.get_logger().info("[state] → PREDICT")

        if self.state == "PREDICT":
            rgb = (self.rgb_first.astype(np.float32) / 255.0)
            x = torch.from_numpy(rgb).permute(2,0,1).unsqueeze(0).to(self.device)
            out = self.model(x)[0].cpu().numpy()  # [2,6] in [-1,1] (가정)
            q1, q2 = out[0], out[1]
            if self.arm_min is not None:
                q1 = denorm_minmax(q1, self.arm_min, self.arm_max)
                q2 = denorm_minmax(q2, self.arm_min, self.arm_max)
            self.q1 = q1.astype(np.float32); self.q2 = q2.astype(np.float32)
            self.get_logger().info(f"[predict] q1={np.round(self.q1,3)}  q2={np.round(self.q2,3)}")

            self._wait_for_traj_connection(timeout=5.0)
            self._send_arm(self.q1, self.move_time)
            self.reach_counter = 0
            self.state = "WAIT_GRASP_POSE"; self.state_ts = now

        elif self.state == "WAIT_GRASP_POSE":
            if self._reached(self.q1, self.q_current6, self.reach_tol): self.reach_counter += 1
            else: self.reach_counter = 0
            if self.reach_counter >= self.reach_stable_count or (now - self.state_ts) > self.reach_timeout:
                if self._send_gripper(self.grip_close): self.get_logger().info("[gripper] closing")
                else: self.get_logger().warn("[gripper] skip close (no server)")
                self.state = "WAIT_GRIP_CLOSE"; self.state_ts = now

        elif self.state == "WAIT_GRIP_CLOSE":
            if (now - self.state_ts) >= self.grip_timeout:
                # === 새 선택 규칙: q2의 FK(xyz)에서 y좌표 부호로 좌/우 트리 결정 ===
                xyz_q2 = fk_xyz(self.q2)  # q2 → FK
                y2 = float(xyz_q2[1])
                if y2 > self.fk_y_epsilon:
                    forest_key = "right"
                elif y2 < -self.fk_y_epsilon:
                    forest_key = "left"
                else:
                    # y≈0이면 q1과 더 가까운 root를 가진 숲으로 타이브레이크
                    xyz_q1 = fk_xyz(self.q1)
                    rootL = next(k for k, v in PARENT_LEFT.items() if v is None)
                    rootR = next(k for k, v in PARENT_RIGHT.items() if v is None)
                    dL = np.linalg.norm(fk_xyz(NODES_Q[rootL]) - xyz_q1)
                    dR = np.linalg.norm(fk_xyz(NODES_Q[rootR]) - xyz_q1)
                    forest_key = "left" if dL <= dR else "right"

                self.tree = self.forest[forest_key]
                self.get_logger().info(f"[tree] selected={forest_key} by q2 FK y={y2:.4f}")

                # --- 가장 가까운 노드( q1의 FK 기준 )을 선택하고 root까지 등반 ---
                xyz_q1 = fk_xyz(self.q1)
                near_key = self.tree.nearest_by_xyz(xyz_q1)
                self.climb_path = self.tree.path_to_root(near_key)  # [start..root]
                self.climb_idx = 0
                self.get_logger().info(f"[tree] nearest={near_key}, path={self.climb_path}")

                q_goal = self.tree.nodes_q[self.climb_path[self.climb_idx]]
                self._send_arm(q_goal, self.move_time)
                self.reach_counter = 0
                self.state = "CLIMB_TO_ROOT"
                self.state_ts = now

        elif self.state == "CLIMB_TO_ROOT":
            q_goal = self.tree.nodes_q[self.climb_path[self.climb_idx]]
            if self._reached(q_goal, self.q_current6, self.reach_tol): self.reach_counter += 1
            else: self.reach_counter = 0
            if self.reach_counter >= self.reach_stable_count or (now - self.state_ts) > self.reach_timeout:
                self.climb_idx += 1
                if self.climb_idx < len(self.climb_path):
                    q_goal = self.tree.nodes_q[self.climb_path[self.climb_idx]]
                    self._send_arm(q_goal, self.move_time)
                    self.reach_counter = 0; self.state_ts = now
                    self.get_logger().info(f"[tree] next → {self.climb_path[self.climb_idx]}")
                else:
                    self._send_arm(self.q2, self.move_time)
                    self.reach_counter = 0
                    self.state = "WAIT_RELEASE_POSE"; self.state_ts = now
                    self.get_logger().info("[state] → WAIT_RELEASE_POSE")

        elif self.state == "WAIT_RELEASE_POSE":
            if self._reached(self.q2, self.q_current6, self.reach_tol): self.reach_counter += 1
            else: self.reach_counter = 0
            if self.reach_counter >= self.reach_stable_count or (now - self.state_ts) > self.reach_timeout:
                if self._send_gripper(self.grip_open): self.get_logger().info("[gripper] opening")
                else: self.get_logger().warn("[gripper] skip open (no server)")
                self.state = "WAIT_GRIP_OPEN"; self.state_ts = now

        elif self.state == "WAIT_GRIP_OPEN":
            if (now - self.state_ts) >= self.grip_timeout:
                self.state = "DONE"; self.get_logger().info("[state] → DONE")

        elif self.state == "DONE":
            pass

    # ---- low level ----
    def _reached(self, target: np.ndarray, current: np.ndarray, tol: float) -> bool:
        err = np.abs(angle_wrap_diff(current, target))
        return bool(np.all(err <= tol))

    def _send_arm(self, q6: np.ndarray, move_time: float):
        msg = JointTrajectory()
        msg.header = Header()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.joint_names = self.joint_names
        pt = JointTrajectoryPoint()
        pt.positions = [float(x) for x in q6.tolist()]
        t = float(move_time)
        pt.time_from_start.sec = int(t); pt.time_from_start.nanosec = int((t - int(t)) * 1e9)
        msg.points.append(pt)
        self.pub_traj.publish(msg)
        self.get_logger().info(
            f"[traj] sent {len(msg.points)}pt to {self.traj_topic} "
            f"names={msg.joint_names} "
            f"pos={np.round(np.array(pt.positions), 3).tolist()} "
            f"T={move_time}s"
        )

    def _send_gripper(self, position: float) -> bool:
        if self.grip_client is None: return False
        if not self.grip_client.wait_for_server(timeout_sec=0.0): return False
        goal = GripperCommandAction.Goal()
        goal.command.position = float(np.clip(position, 0.0, 0.8))
        goal.command.max_effort = 0.0
        self.grip_client.send_goal_async(goal)
        return True


def main():
    rclpy.init()
    node = KeyframeTreeFKRunner()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__":
    main()
