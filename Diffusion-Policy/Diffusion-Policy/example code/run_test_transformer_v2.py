#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import time
from collections import deque
from typing import List, Optional

import numpy as np
import rclpy
from rclpy.node import Node

from sensor_msgs.msg import Image, JointState
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from control_msgs.msg import GripperCommand
from std_msgs.msg import Header

import torch
import torch.nn as nn
import cv2

try:
    from cv_bridge import CvBridge
except Exception as e:
    CvBridge = None

# ===== 프로젝트 임포트 (경로 맞춤) =====
# 모델: Transformer (dq 없이 RGB+Depth+q -> 미래 q)
from models.transformer_joint_gripper_no_dq import TransformerJointGripperNoDQ


def to_numpy_image(msg: Image) -> np.ndarray:
    """ROS Image -> numpy (RGB 또는 Depth)."""
    if CvBridge is not None:
        cvb = CvBridge()
        if msg.encoding in ("rgb8", "bgr8"):
            cv_img = cvb.imgmsg_to_cv2(msg, desired_encoding="bgr8")
            rgb = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
            return rgb.astype(np.uint8)
        elif msg.encoding in ("mono16", "16UC1", "32FC1"):
            depth = cvb.imgmsg_to_cv2(msg)
            return np.asarray(depth)
        else:
            # fallback: try convert to bgr8
            cv_img = cvb.imgmsg_to_cv2(msg, desired_encoding="passthrough")
            if cv_img.ndim == 2:
                return cv_img
            rgb = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
            return rgb
    else:
        raise RuntimeError("cv_bridge not available. Install ros-<distro>-cv-bridge.")


def minmax_norm(x: torch.Tensor, mn: np.ndarray, mx: np.ndarray) -> torch.Tensor:
    mn_t = torch.as_tensor(mn, dtype=x.dtype, device=x.device)
    mx_t = torch.as_tensor(mx, dtype=x.dtype, device=x.device)
    rng = torch.clamp(mx_t - mn_t, min=1e-6)
    return 2.0 * (x - mn_t) / rng - 1.0


def minmax_denorm(x: torch.Tensor, mn: np.ndarray, mx: np.ndarray) -> torch.Tensor:
    mn_t = torch.as_tensor(mn, dtype=x.dtype, device=x.device)
    mx_t = torch.as_tensor(mx, dtype=x.dtype, device=x.device)
    return (x + 1.0) * 0.5 * (mx_t - mn_t) + mn_t


class PolicyRunner(Node):
    """
    - RGB/Depth/JointStates를 받아 obs_horizon=K 슬라이딩 윈도우 구성
    - Transformer로 [T,7] (정규화) 예측 → 디노멀라이즈 → 첫 스텝만 퍼블리시
    - Kinova Gen3 Lite: joint_1..joint_6 + gripper
    """

    def __init__(self):
        super().__init__("policy_runner")

        # ----------------- 파라미터 -----------------
        self.declare_parameter("model_path", "")
        self.declare_parameter("stats_path", "")
        self.declare_parameter("joint_names", ["joint_1","joint_2","joint_3","joint_4","joint_5","joint_6"])
        self.declare_parameter("gripper_joint_name", "right_finger_bottom_joint")  # 환경에 맞게!
        self.declare_parameter("gripper_in_trajectory", True)  # True면 7개를 하나의 JointTrajectory로
        self.declare_parameter("traj_topic", "/joint_trajectory_controller/joint_trajectory")
        self.declare_parameter("gripper_topic", "/gripper_controller/command")
        self.declare_parameter("color_topic", "/camera/color/image_raw")
        self.declare_parameter("depth_topic", "/camera/depth/image_rect_raw")
        self.declare_parameter("rate_hz", 5.0)
        self.declare_parameter("move_time", 0.2)  # sec
        self.declare_parameter("obs_horizon", 8)
        self.declare_parameter("pred_horizon", 12)
        self.declare_parameter("device", "cuda")
        self.declare_parameter("img_h", 240)
        self.declare_parameter("img_w", 424)
        self.declare_parameter("use_tanh_out", True)
        self.declare_parameter("dq_step_clip", 0.0)   # 절대 q 예측이므로 보통 0.0 유지(미사용)
        self.declare_parameter("q_limits", [])        # [[lo,hi],...]*6 (미설정시 미사용)
        self.declare_parameter("print_debug", False)

        # 파라미터 로드
        self.model_path: str = self.get_parameter("model_path").get_parameter_value().string_value
        self.stats_path: str = self.get_parameter("stats_path").get_parameter_value().string_value
        self.joint_names: List[str] = list(self.get_parameter("joint_names").get_parameter_value().string_array_value)
        self.gripper_joint_name: str = self.get_parameter("gripper_joint_name").get_parameter_value().string_value
        self.gripper_in_trajectory: bool = self.get_parameter("gripper_in_trajectory").get_parameter_value().bool_value
        self.traj_topic: str = self.get_parameter("traj_topic").get_parameter_value().string_value
        self.gripper_topic: str = self.get_parameter("gripper_topic").get_parameter_value().string_value
        self.color_topic: str = self.get_parameter("color_topic").get_parameter_value().string_value
        self.depth_topic: str = self.get_parameter("depth_topic").get_parameter_value().string_value
        self.rate_hz: float = float(self.get_parameter("rate_hz").get_parameter_value().double_value)
        self.move_time: float = float(self.get_parameter("move_time").get_parameter_value().double_value)
        self.K: int = int(self.get_parameter("obs_horizon").get_parameter_value().integer_value)
        self.T: int = int(self.get_parameter("pred_horizon").get_parameter_value().integer_value)
        self.device_str: str = self.get_parameter("device").get_parameter_value().string_value
        self.img_h: int = int(self.get_parameter("img_h").get_parameter_value().integer_value)
        self.img_w: int = int(self.get_parameter("img_w").get_parameter_value().integer_value)
        self.use_tanh_out: bool = self.get_parameter("use_tanh_out").get_parameter_value().bool_value
        self.dq_step_clip: float = float(self.get_parameter("dq_step_clip").get_parameter_value().double_value)
        self.q_limits = self.get_parameter("q_limits").get_parameter_value().double_array_value
        self.print_debug: bool = self.get_parameter("print_debug").get_parameter_value().bool_value

        # 버퍼
        self.rgb_buf  = deque(maxlen=self.K)   # [H,W,3] uint8
        self.dep_buf  = deque(maxlen=self.K)   # [H,W] float32
        self.q_buf    = deque(maxlen=self.K)   # [7] float32
        self.q_cur    = np.zeros(7, dtype=np.float32)

        # 퍼블리셔
        self.pub_traj = self.create_publisher(JointTrajectory, self.traj_topic, 10)
        self.pub_grip = None
        if not self.gripper_in_trajectory:
            self.pub_grip = self.create_publisher(GripperCommand, self.gripper_topic, 10)

        # 서브스크라이버
        self.sub_js   = self.create_subscription(JointState, "/joint_states", self.cb_joint_states, 10)
        self.sub_rgb  = self.create_subscription(Image, self.color_topic, self.cb_rgb, 10)
        self.sub_dep  = self.create_subscription(Image, self.depth_topic, self.cb_depth, 10)

        # Torch 설정
        self.device = torch.device(self.device_str if torch.cuda.is_available() and self.device_str == "cuda" else "cpu")
        self.model = self._load_model(self.model_path)
        self.stats = self._load_stats(self.stats_path)

        # 타이머 루프
        self.timer = self.create_timer(1.0 / max(1e-3, self.rate_hz), self.tick)
        self.get_logger().info(f"[policy_runner] ready. device={self.device}, hz={self.rate_hz}")

    # ------------ 로더들 ------------
    def _load_model(self, path: str) -> nn.Module:
        if not path or not os.path.exists(path):
            raise FileNotFoundError(f"model_path not found: {path}")

        # param npz도 같이 넘겨주세요: --param_path ~/.../transformer_joint_abs_2_param.npz
        param_path = self.get_parameter("param_path").get_parameter_value().string_value
        if not param_path or not os.path.exists(param_path):
            raise FileNotFoundError(f"param_path not found: {param_path}")

        P = np.load(param_path)
        model = TransformerJointGripperNoDQ(
            d_model=int(P["d_model"]),
            nhead=int(P["nhead"]),
            num_encoder_layers=int(P["tfm_layers"]),
            num_decoder_layers=int(P["tfm_layers"]),
            dim_feedforward=int(P["dim_ff"]),
            dropout=float(P["tfm_dropout"]),
            img_feat_dim=int(P["img_feat_dim"]),
            q_feat_dim=int(P["q_feat_dim"]),
            pred_horizon=int(P["pred_horizon"]),
            out_dim=int(P["out_dim"]),
            use_tanh_out=self.use_tanh_out,
        ).to(self.device)

        state = torch.load(path, map_location="cpu")
        model.load_state_dict(state)  # EMA로 저장된 가중치라 그대로 OK
        model.eval()
        return model

    def _load_stats(self, path: str):
        if path is None or path == "" or not os.path.exists(path):
            raise FileNotFoundError(f"stats_path not found: {path}")
        z = np.load(path, allow_pickle=False)
        # ROS-safe 저장 포맷: stats__q__min, stats__q__max
        q_min = z["stats__q__min"]
        q_max = z["stats__q__max"]
        return {"q": {"min": q_min, "max": q_max}}

    # ------------ 콜백 ------------
    def cb_joint_states(self, msg: JointState):
        name_to_idx = {n: i for i, n in enumerate(msg.name)}
        q = np.zeros(7, dtype=np.float32)

        # 6 관절
        for j, n in enumerate(self.joint_names):
            if n in name_to_idx:
                q[j] = float(msg.position[name_to_idx[n]])
        # 그리퍼
        if self.gripper_joint_name in name_to_idx:
            q[-1] = float(msg.position[name_to_idx[self.gripper_joint_name]])

        self.q_cur = q
        self.q_buf.append(q)

    def cb_rgb(self, msg: Image):
        try:
            arr = to_numpy_image(msg)  # uint8 RGB
            if arr.ndim == 3 and arr.shape[2] == 3:
                if (arr.shape[0] != self.img_h) or (arr.shape[1] != self.img_w):
                    arr = cv2.resize(arr, (self.img_w, self.img_h), interpolation=cv2.INTER_AREA)
                self.rgb_buf.append(arr)
        except Exception as e:
            self.get_logger().warn(f"RGB parse error: {e}")

    def cb_depth(self, msg: Image):
        try:
            arr = to_numpy_image(msg)  # float/uint depth
            if arr.ndim == 2:
                if (arr.shape[0] != self.img_h) or (arr.shape[1] != self.img_w):
                    arr = cv2.resize(arr, (self.img_w, self.img_h), interpolation=cv2.INTER_NEAREST)
                self.dep_buf.append(arr.astype(np.float32))
        except Exception as e:
            self.get_logger().warn(f"Depth parse error: {e}")

    # ------------ 메인 루프 ------------
    @torch.no_grad()
    def tick(self):
        if len(self.rgb_buf) < self.K or len(self.dep_buf) < self.K or len(self.q_buf) < self.K:
            return

        # 입력 구성 (B=1)
        rgb_seq = np.stack(list(self.rgb_buf)[-self.K:], axis=0)    # [K,H,W,3] uint8
        dep_seq = np.stack(list(self.dep_buf)[-self.K:], axis=0)    # [K,H,W] float32 (meters or scaled)
        q_seq   = np.stack(list(self.q_buf)[-self.K:], axis=0)      # [K,7] float32

        # --- 정규화/텐서 변환 ---
        # RGB: 0~1, [B,K,3,H,W]
        rgb = torch.from_numpy(rgb_seq.astype(np.float32) / 255.0).permute(0,3,1,2).unsqueeze(0).to(self.device)
        # Depth: 0~1로 스케일 (필요시 max_depth_m로 나누는 전처리 추가)
        # 여기서는 depth가 미터라 가정, 0~2.5m 클립 후 /2.5
        dep = dep_seq.astype(np.float32)
        # 퍼센타일 클립 (프레임/배치 단위로 적용)
        lo = np.percentile(dep, 1.0)
        hi = np.percentile(dep, 99.0)
        dep = np.clip(dep, lo, hi)
        # 1.0 m 기준 정규화
        dep = np.clip(dep, 0.0, 1.0)  # max_depth_m=1.0 가정
        depth = torch.from_numpy(dep).unsqueeze(1).unsqueeze(0).to(self.device)  # [1,K,1,H,W]

        # q: [-1,1] 정규화
        q_mn = self.stats["q"]["min"]; q_mx = self.stats["q"]["max"]
        q_t  = torch.from_numpy(q_seq).unsqueeze(0).to(self.device)  # [1,K,7]
        q_n  = minmax_norm(q_t, q_mn, q_mx)

        # --- 모델 추론 ---
        pred_n = self.model(obs_rgb=rgb, obs_depth=depth, obs_q=q_n, pred_horizon=self.T)  # [1,T,7] [-1,1]
        # 디노멀라이즈 → 절대 q
        pred_abs = minmax_denorm(pred_n, q_mn, q_mx)[0].cpu().numpy()  # [T,7]
        q_cmd = pred_abs[0]  # 첫 스텝

        # (옵션) joint limit clip
        if self.q_limits:
            q_cmd[:6] = self._clip_q_limits(q_cmd[:6])

        # (옵션) per-step Δq clip — 절대 q에서는 미사용이 일반적
        if self.dq_step_clip > 0.0:
            dq = np.clip(q_cmd[:6] - self.q_cur[:6], -self.dq_step_clip, self.dq_step_clip)
            q_cmd[:6] = self.q_cur[:6] + dq

        # 퍼블리시
        self._publish(q_cmd)

        if self.print_debug:
            self.get_logger().info(f"cmd: {np.array2string(q_cmd, precision=3)}")

    # ------------ 유틸 ------------
    def _clip_q_limits(self, q6: np.ndarray) -> np.ndarray:
        """q_limits=[[lo,hi],...]*6 가 주어졌을 때 클립."""
        q6c = q6.copy()
        try:
            L = np.array(self.q_limits, dtype=np.float32).reshape(-1,2)
            for i in range(min(6, L.shape[0])):
                lo, hi = float(L[i,0]), float(L[i,1])
                q6c[i] = np.clip(q6c[i], lo, hi)
        except Exception:
            pass
        return q6c

    def _publish(self, q7: np.ndarray):
        """
        q7: [6 + grip], 절대 위치 명령
        - gripper_in_trajectory=True: 7개를 한 번에 JointTrajectory로
        - False: 6개는 JointTrajectory, 그리퍼는 GripperCommand
        """
        if self.gripper_in_trajectory:
            msg = JointTrajectory()
            msg.header = Header()
            msg.header.stamp = self.get_clock().now().to_msg()
            msg.joint_names = self.joint_names + [self.gripper_joint_name]
            pt = JointTrajectoryPoint()
            pt.positions = list(q7.astype(float))
            pt.time_from_start.sec = int(self.move_time)
            pt.time_from_start.nanosec = int((self.move_time - int(self.move_time)) * 1e9)
            msg.points.append(pt)
            self.pub_traj.publish(msg)
        else:
            # Arm 6
            msg = JointTrajectory()
            msg.header = Header()
            msg.header.stamp = self.get_clock().now().to_msg()
            msg.joint_names = self.joint_names
            pt = JointTrajectoryPoint()
            pt.positions = list(q7[:6].astype(float))
            pt.time_from_start.sec = int(self.move_time)
            pt.time_from_start.nanosec = int((self.move_time - int(self.move_time)) * 1e9)
            msg.points.append(pt)
            self.pub_traj.publish(msg)

            # Gripper (0.0 ~ 0.8 절대 위치)
            if self.pub_grip is not None:
                gmsg = GripperCommand()
                gmsg.position = float(np.clip(q7[-1], 0.0, 0.8))
                gmsg.max_effort = 0.0  # 컨트롤러 기본값 사용 (필요 시 세팅)
                self.pub_grip.publish(gmsg)


def main():
    rclpy.init()
    node = PolicyRunner()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    except Exception as e:
        node.get_logger().error(f"Fatal error: {e}")
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
