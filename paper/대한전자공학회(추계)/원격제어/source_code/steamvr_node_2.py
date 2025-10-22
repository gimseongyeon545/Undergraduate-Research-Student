#!/usr/bin/env python3
# 파일명: steamvr_node_2.py
import math
import numpy as np
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
import openvr
from sensor_msgs.msg import Joy


def mat34_to_pose(m):
    """
    OpenVR 3x4 행렬을 (t, qx,qy,qz,qw)로 변환.
    - 회전행렬 SVD 정규화
    - 수치 안전 쿼터니언 변환
    - NaN/Inf/노름≈0 이면 None 반환(퍼블리시 스킵)
    """
    R = np.array([[m[0][0], m[0][1], m[0][2]],
                  [m[1][0], m[1][1], m[1][2]],
                  [m[2][0], m[2][1], m[2][2]]], dtype=float)
    t = np.array([m[0][3], m[1][3], m[2][3]], dtype=float)

    # 가장 가까운 회전행렬로 보정
    U, _, Vt = np.linalg.svd(R)
    R = U @ Vt
    if np.linalg.det(R) < 0.0:
        Vt[-1, :] *= -1.0
        R = U @ Vt

    tr = float(np.trace(R))
    tr = max(min(tr, 3.0), -1.0)  # trace 안전 클램프

    if tr > 0.0:
        S = math.sqrt(tr + 1.0) * 2.0
        qw = 0.25 * S
        qx = (R[2, 1] - R[1, 2]) / S
        qy = (R[0, 2] - R[2, 0]) / S
        qz = (R[1, 0] - R[0, 1]) / S
    else:
        if (R[0, 0] > R[1, 1]) and (R[0, 0] > R[2, 2]):
            S = math.sqrt(max(1.0 + R[0, 0] - R[1, 1] - R[2, 2], 1e-12)) * 2.0
            qx = 0.25 * S
            qy = (R[0, 1] + R[1, 0]) / S
            qz = (R[0, 2] + R[2, 0]) / S
            qw = (R[2, 1] - R[1, 2]) / S
        elif R[1, 1] > R[2, 2]:
            S = math.sqrt(max(1.0 + R[1, 1] - R[0, 0] - R[2, 2], 1e-12)) * 2.0
            qx = (R[0, 1] + R[1, 0]) / S
            qy = 0.25 * S
            qz = (R[1, 2] + R[2, 1]) / S
            qw = (R[0, 2] - R[2, 0]) / S
        else:
            S = math.sqrt(max(1.0 + R[2, 2] - R[0, 0] - R[1, 1], 1e-12)) * 2.0
            qx = (R[0, 2] + R[2, 0]) / S
            qy = (R[1, 2] + R[2, 1]) / S
            qz = 0.25 * S
            qw = (R[1, 0] - R[0, 1]) / S

    q = np.array([qx, qy, qz, qw], dtype=float)
    n = np.linalg.norm(q)

    if not np.isfinite(t).all() or not np.isfinite(q).all() or n < 1e-9:
        return None

    q /= n
    return t, (q[0], q[1], q[2], q[3])


class SteamVRNode2(Node):
    def __init__(self):
        super().__init__('steamvr_node_2')

        # ⚠️ Node에 publishers라는 read-only 프로퍼티가 있어서 이름 충돌 방지
        self._pose_pubs = {
    "left":  self.create_publisher(PoseStamped, '/steamvr/left', 10),
    "right": self.create_publisher(PoseStamped, '/steamvr/right', 10),
}
       
        # OpenVR 초기화
        openvr.init(openvr.VRApplication_Background)
        self.vr = openvr.VRSystem()
        self.get_logger().info('OpenVR initialized.')

        # 100Hz 타이머
        self.timer = self.create_timer(1.0 / 100.0, self.tick)
        
        
        self._joy_pubs = {
    "left":  self.create_publisher(Joy, '/steamvr/left/joy', 10),
    "right": self.create_publisher(Joy, '/steamvr/right/joy', 10),
}

        
        # 마지막 전송값 저장(변화 감지용 디버그)
        self._last_buttons = [0, 0]
        self._last_axes0 = [0.0, 0.0]   # controller0: [pad_x, pad_y]
        self._last_axes1 = [0.0, 0.0]   # controller1: [pad_x, pad_y]

    def tick(self):
        try:
            poses_t = openvr.TrackedDevicePose_t * openvr.k_unMaxTrackedDeviceCount
            tracked = poses_t()

            self.vr.getDeviceToAbsoluteTrackingPose(
            openvr.TrackingUniverseStanding, 0.0, tracked
        )
            now = self.get_clock().now().to_msg()

            # === 역할 기반 인덱스 ===
            left_idx  = self.vr.getTrackedDeviceIndexForControllerRole(
            openvr.TrackedControllerRole_LeftHand
        )
            right_idx = self.vr.getTrackedDeviceIndexForControllerRole(
            openvr.TrackedControllerRole_RightHand
        )

            pairs = [
            ("left", left_idx),
            ("right", right_idx),
        ]

            for side, dev_idx in pairs:
                if dev_idx == openvr.k_unTrackedDeviceIndexInvalid:
                    continue
                pose = tracked[dev_idx]
                if not pose.bPoseIsValid:
                    continue

                m = pose.mDeviceToAbsoluteTracking
                res = mat34_to_pose(m)
                if res is None:
                    continue

                t, q = res
                msg = PoseStamped()
                msg.header.stamp = now
                msg.header.frame_id = 'steamvr_world'
                msg.pose.position.x, msg.pose.position.y, msg.pose.position.z = t.tolist()
                msg.pose.orientation.x, msg.pose.orientation.y, msg.pose.orientation.z, msg.pose.orientation.w = q

                # 퍼블리셔를 역할별로 분리 (초기화에서 준비해두기)
                self._pose_pubs[side].publish(msg)

                joy = self._read_inputs(dev_idx)
                if joy is not None:
                    axes, buttons = joy
                    msgj = Joy()
                    msgj.header.stamp = now
                    msgj.axes = axes
                    msgj.buttons = buttons
                    self._joy_pubs[side].publish(msgj)

        except Exception as e:
            self.get_logger().error(f'SteamVR tick failed: {e}')






    def destroy_node(self):
        try:
            openvr.shutdown()
        except Exception:
            pass
        super().destroy_node()
        
    def _read_inputs(self, dev_idx):
        """OpenVR controller state -> (axes, buttons)
        axes: [pad_x, pad_y, trigger]
        buttons: [system, menu, grip, trigger_click, pad_click, pad_touch]
        """
        try:
            ok, state = self.vr.getControllerState(dev_idx)
        except Exception:
            return None

        if not ok:
            return None

        # 축: Vive wand 관례 (컨트롤러에 따라 달라질 수 있음)
        # rAxis[0]: trackpad (x,y), rAxis[1]: trigger (x), rAxis[?].y는 쓰지 않음
        pad_x = float(state.rAxis[0].x) if len(state.rAxis) > 0 else 0.0
        pad_y = float(state.rAxis[0].y) if len(state.rAxis) > 0 else 0.0
        trig  = float(state.rAxis[1].x) if len(state.rAxis) > 1 else 0.0

        axes = [pad_x, pad_y, trig]

        # 버튼 비트 (OpenVR EVRButtonId 관례)
        # System=0, ApplicationMenu=1, Grip=2, Axis0(Pad)=32, Axis1(Trigger)=33
        pressed = state.ulButtonPressed
        touched = state.ulButtonTouched

        def bit(b): return 1 if (pressed & (1 << b)) else 0
        def touchbit(b): return 1 if (touched & (1 << b)) else 0

        system_btn   = bit(0)
        menu_btn     = bit(1)  # application/menu
        grip_btn     = bit(2)
        trig_click   = bit(33)  # 완전 당겼을 때 click 비트
        pad_click    = bit(32)
        pad_touch    = touchbit(32)

        buttons = [system_btn, menu_btn, grip_btn, trig_click, pad_click, pad_touch]
        return axes, buttons



def main():
    rclpy.init()
    node = SteamVRNode2()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()

