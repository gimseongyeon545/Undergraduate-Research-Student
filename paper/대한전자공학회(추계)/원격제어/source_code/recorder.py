#!/usr/bin/env python3
import os
import re
import signal
import math
import numpy as np
from collections import defaultdict, deque
from typing import Optional, Tuple

import rclpy
import math
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg import Image, JointState, CameraInfo
from std_msgs.msg import Header
from cv_bridge import CvBridge
import message_filters

import tf2_ros
from geometry_msgs.msg import TransformStamped
from std_msgs.msg import Bool

def next_available_path(path: str) -> str:
    """
    path가 존재하면 숫자 접미사를 자동 증가시켜
    존재하지 않는 경로를 반환한다.
    규칙:
      - 파일명이 ...NNN 형태면 그 NNN부터 올림 (자릿수 유지)
      - 숫자 없으면 _001부터 붙임
    예)
      scenario_001.npz -> scenario_002.npz ...
      scenario.npz     -> scenario_001.npz ...
    """
    d, fname = os.path.split(path)
    root, ext = os.path.splitext(fname)
    m = re.search(r'^(.*?)(\d+)$', root)
    if m:
        prefix, num_str = m.group(1), m.group(2)
        n = int(num_str)
        width = len(num_str)
    else:
        prefix = root + "_"
        n = 1
        width = 4

    while True:
        candidate = os.path.join(d, f"{prefix}{n:0{width}d}{ext}")
        if not os.path.exists(candidate):
            return candidate
        n += 1

def euler_from_quaternion(q):
    """q = [x, y, z, w] → (roll, pitch, yaw) 라디안 반환."""
    x, y, z, w = q
    # roll (x-axis rotation)
    t0 = +2.0 * (w * x + y * z)
    t1 = +1.0 - 2.0 * (x * x + y * y)
    roll = math.atan2(t0, t1)

    # pitch (y-axis rotation)
    t2 = +2.0 * (w * y - z * x)
    t2 = +1.0 if t2 > +1.0 else t2
    t2 = -1.0 if t2 < -1.0 else t2
    pitch = math.asin(t2)

    # yaw (z-axis rotation)
    t3 = +2.0 * (w * z + x * y)
    t4 = +1.0 - 2.0 * (y * y + z * z)
    yaw = math.atan2(t3, t4)
    return (roll, pitch, yaw)

def depth_to_float_meters(msg: Image):
    """Depth Image를 float32 [m] numpy array로 변환 (16UC1/32FC1 대응)."""
    import numpy as np
    import cv2
    if msg.encoding in ("16UC1", "mono16"):
        depth_raw = np.frombuffer(msg.data, dtype=np.uint16).reshape((msg.height, msg.width))
        return depth_raw.astype(np.float32) / 1000.0  # mm -> m (RealSense 일반)
    elif msg.encoding in ("32FC1",):
        depth_raw = np.frombuffer(msg.data, dtype=np.float32).reshape((msg.height, msg.width))
        return depth_raw
    else:
        # cv_bridge로 안전 변환 시도
        br = CvBridge()
        cvimg = br.imgmsg_to_cv2(msg, desired_encoding="passthrough")
        if cvimg.dtype == np.uint16:
            return cvimg.astype(np.float32)/1000.0
        return cvimg.astype(np.float32)

class SyncRecorder(Node):
    def __init__(self):
        super().__init__("gen3lite_sync_recorder")

        # 파라미터
        self.declare_parameter("rgb_topic", "/camera/color/image_raw")
        self.declare_parameter("depth_topic", "/camera/depth/image_rect_raw")
        self.declare_parameter("joint_topic", "/joint_states")
        self.declare_parameter("camera_info_topic", "/camera/color/camera_info")  # intrinsics 저장용(선택)
        self.declare_parameter("base_frame", "base_link")
        self.declare_parameter("ee_frame", "end_effector_link")
        self.declare_parameter("output", "session_0001.npz")
        self.declare_parameter("sync_slop_sec", 0.033)   # 33ms 정도 허용
        self.declare_parameter("queue_size", 30)
        self.declare_parameter("max_samples", 0)         # 0이면 무제한

        self.declare_parameter("gate_by_vr_active", True)
        self.declare_parameter("active_timeout_sec", 0.08)  # ★ 선언 추가
        self.gate_by_vr_active = bool(self.get_parameter("gate_by_vr_active").value)
        self.active_timeout_sec = float(self.get_parameter("active_timeout_sec").value)
        self._last_active_time = -1.0
        self.active_sub = self.create_subscription(Bool, '/vr_active', self.cb_active, 10)
        keys = [
            "rgb_topic", "depth_topic", "joint_topic", "camera_info_topic",
            "base_frame", "ee_frame", "output", "sync_slop_sec", "queue_size", "max_samples"
        ]
        self.cfg = {k: self.get_parameter(k).value for k in keys}
        # ↓↓↓ 추가: output 자동증가 적용
        orig_out = self.cfg["output"]
        out_dir = os.path.dirname(orig_out)
        if out_dir and not os.path.exists(out_dir):
            os.makedirs(out_dir, exist_ok=True)
        # 이미 존재하면 다음 번호로 굴림
        if os.path.exists(orig_out):
            new_out = next_available_path(orig_out)
            self.get_logger().warn(f"[output] {orig_out} 가 존재합니다. 다음 파일명으로 저장합니다: {new_out}")
            self.cfg["output"] = new_out
        else:
            # 숫자 없는 베이스 이름을 준 경우에도 자동 증가를 원하면 아래 한 줄을 주석 해제하세요.
            # self.cfg["output"] = next_available_path(orig_out)
            pass

        self.get_logger().info(f"[CFG] {self.cfg}")

        # QoS (sensor_data)
        sensor_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=10
        )

        # Subscribers with message_filters
        self.rgb_sub   = message_filters.Subscriber(self, Image, self.cfg["rgb_topic"], qos_profile=sensor_qos)
        self.depth_sub = message_filters.Subscriber(self, Image, self.cfg["depth_topic"], qos_profile=sensor_qos)
        self.joint_sub = message_filters.Subscriber(self, JointState, self.cfg["joint_topic"], qos_profile=QoSProfile(depth=50))

        self.sync = message_filters.ApproximateTimeSynchronizer(
            [self.rgb_sub, self.depth_sub, self.joint_sub],
            queue_size=int(self.cfg["queue_size"]),
            slop=float(self.cfg["sync_slop_sec"])
        )
        self.sync.registerCallback(self.cb_sync)

        # CameraInfo (선택 저장)
        self.caminfo_sub = self.create_subscription(CameraInfo, self.cfg["camera_info_topic"], self.cb_caminfo, 10)
        self._camera_info: Optional[CameraInfo] = None

        # TF
        self.tf_buffer = tf2_ros.Buffer(cache_time=rclpy.duration.Duration(seconds=10.0))
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        self.bridge = CvBridge()

        # 버퍼
        self.data = defaultdict(list)
        self.sample_count = 0
        self.max_samples = int(self.cfg["max_samples"])

        # SIGINT 처리
        # signal.signal(signal.SIGINT, self._sigint_handler)

        self.get_logger().info("Sync recorder ready. Press Ctrl+C to save and exit.")

    def cb_active(self, msg: Bool):
        if msg.data:
            self._last_active_time = self.get_clock().now().nanoseconds * 1e-9

    def cb_caminfo(self, msg: CameraInfo):
        # 첫 CamInfo만 저장
        if self._camera_info is None:
            self._camera_info = msg
            self.get_logger().info(f"CameraInfo captured (K={list(msg.k)})")

    def cb_sync(self, rgb_msg: Image, depth_msg: Image, joint_msg: JointState):
        if self.gate_by_vr_active:
            now = self.get_clock().now().nanoseconds * 1e-9
            if (self._last_active_time < 0.0) or (now - self._last_active_time > self.active_timeout_sec):
                return  # VR 비활성 → 저장 스킵
        # 공통 타임스탬프(ns)
        stamp_ns = rgb_msg.header.stamp.sec * 10**9 + rgb_msg.header.stamp.nanosec

        # RGB → np.uint8(H,W,3)
        try:
            rgb = self.bridge.imgmsg_to_cv2(rgb_msg, desired_encoding="bgr8")[..., ::-1].copy()
        except Exception as e:
            self.get_logger().warn(f"cv_bridge rgb convert failed: {e}")
            return

        try:
            depth = depth_to_float_meters(depth_msg)
        except Exception as e:
            self.get_logger().warn(f"depth convert failed: {e}")
            return

        # ★ 여기서 1회만 실제 해상도 로그
        if self.sample_count == 0:
            self.get_logger().info(f"RGB shape={rgb.shape}, Depth shape={depth.shape}")

        # Joint states
        joint_names = list(joint_msg.name)
        positions = np.array(joint_msg.position, dtype=np.float32)
        velocities = np.array(joint_msg.velocity, dtype=np.float32) if joint_msg.velocity else None

        # EE pose from TF at approx same time
        ee_xyz, ee_rpy = self.lookup_ee_pose(rgb_msg.header)

        # 저장
        self.data["stamp_ns"].append(np.int64(stamp_ns))
        self.data["rgb"].append(rgb)
        self.data["depth"].append(depth)
        self.data["joint_names"] = joint_names  # 매 샘플 동일 가정, 한 번만 저장
        self.data["q"].append(positions)
        if velocities is not None:
            self.data["dq"].append(velocities)
        if ee_xyz is not None:
            self.data["ee_xyz"].append(ee_xyz.astype(np.float32))
            self.data["ee_rpy"].append(ee_rpy.astype(np.float32))

        self.sample_count += 1
        if self.sample_count % 50 == 0:
            self.get_logger().info(f"[{self.sample_count}] samples buffered...")

        # 자동 종료(옵션)
        if self.max_samples > 0 and self.sample_count >= self.max_samples:
            self.get_logger().info("Reached max_samples. Saving...")
            self.save_and_exit()

    def lookup_ee_pose(self, header: Header) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        base = self.cfg["base_frame"]
        ee   = self.cfg["ee_frame"]
        try:
            # header stamp 기준으로 TF 조회 (fail시 최신값 fallback)
            when = rclpy.time.Time(seconds=header.stamp.sec, nanoseconds=header.stamp.nanosec)
            tf: TransformStamped = self.tf_buffer.lookup_transform(
                target_frame=base,
                source_frame=ee,
                time=when,
                timeout=rclpy.duration.Duration(seconds=0.05)
            )
        except Exception:
            try:
                tf = self.tf_buffer.lookup_transform(base, ee, rclpy.time.Time())
            except Exception:
                return None, None

        t = tf.transform.translation
        q = tf.transform.rotation
        xyz = np.array([t.x, t.y, t.z], dtype=np.float64)
        quat = [q.x, q.y, q.z, q.w]
        rpy = np.array(euler_from_quaternion(quat), dtype=np.float64)  # roll, pitch, yaw
        return xyz, rpy

    def _sigint_handler(self, signum, frame):
        self.get_logger().info("Ctrl+C detected. Saving .npz ...")
        self.save_and_exit()

    def save_and_exit(self):
        out = self.cfg["output"]
        os.makedirs(os.path.dirname(out) if os.path.dirname(out) else ".", exist_ok=True)

        N = len(self.data["rgb"])
        if N == 0:
            self.get_logger().error("No synchronized samples captured. Nothing to save.")
            return

        # list → numpy stack
        pack = {}
        pack["stamp_ns"] = np.asarray(self.data["stamp_ns"], dtype=np.int64)
        pack["rgb"]  = np.stack(self.data["rgb"], axis=0).astype(np.uint8)        # [N,H,W,3]
        pack["depth"] = np.stack(self.data["depth"], axis=0).astype(np.float32)   # [N,H,W]
        pack["q"]    = np.stack(self.data["q"], axis=0).astype(np.float32)        # [N,J]
        if "dq" in self.data and len(self.data["dq"])>0:
            pack["dq"] = np.stack(self.data["dq"], axis=0).astype(np.float32)
        if "ee_xyz" in self.data:
            pack["ee_xyz"] = np.stack(self.data["ee_xyz"], axis=0)
        if "ee_rpy" in self.data:
            pack["ee_rpy"] = np.stack(self.data["ee_rpy"], axis=0)

        # joint_names / camera intrinsics 등 메타데이터
        if "joint_names" in self.data:
            pack["joint_names"] = np.array(self.data["joint_names"], dtype=object)
        if self._camera_info is not None:
            # K(3x3), D, width/height 저장
            pack["camera_K"] = np.array(self._camera_info.k, dtype=np.float64).reshape(3,3)
            pack["camera_D"] = np.array(self._camera_info.d, dtype=np.float64)
            pack["camera_width"]  = np.int32(self._camera_info.width)
            pack["camera_height"] = np.int32(self._camera_info.height)

        np.savez_compressed(out, **pack)
        self.get_logger().info(f"Saved: {os.path.abspath(out)} (N={pack['rgb'].shape[0]})")
        # rclpy.shutdown()


def main():
    rclpy.init()
    node = SyncRecorder()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Ctrl+C detected. Saving .npz ...")
    finally:
        try:
            node.save_and_exit()
        except Exception as e:
            node.get_logger().error(f"save failed: {e}")
        node.destroy_node()
        try:
            if rclpy.ok():
                rclpy.shutdown()
        except Exception:
            pass

if __name__ == "__main__":
    main()
