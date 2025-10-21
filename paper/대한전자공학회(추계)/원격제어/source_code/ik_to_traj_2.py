#!/usr/bin/env python3
# 파일명: ik_to_traj_2.py
import math, numpy as np, time
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy
from geometry_msgs.msg import PoseStamped
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from moveit_msgs.srv import GetPositionIK
from sensor_msgs.msg import JointState
import argparse
from builtin_interfaces.msg import Time

# ★ 추가: 그리퍼 액션
from rclpy.action import ActionClient
from control_msgs.action import GripperCommand


def rpy_to_quat(roll, pitch, yaw):
    cy = math.cos(yaw * 0.5); sy = math.sin(yaw * 0.5)
    cp = math.cos(pitch * 0.5); sp = math.sin(pitch * 0.5)
    cr = math.cos(roll * 0.5);  sr = math.sin(roll * 0.5)
    qx = sr * cp * cy - cr * sp * sy
    qy = cr * sp * cy + sr * cp * sy
    qz = cr * cp * sy - sr * sp * cy
    qw = cr * cp * cy + sr * sp * sy
    return qx, qy, qz, qw


class IKToTrajInitAnchor(Node):
    def __init__(self, group_name, ee_link, controller_topic, base_frame,
                 controller_joint_order, anchor_topic, ee_cmd_topic):
        super().__init__('ik_to_traj')
        self.group_name = group_name
        self.ee_link = ee_link
        self.controller_topic = controller_topic
        self.base_frame = base_frame
        self.controller_joint_order = controller_joint_order
        self.joint_state = None

        self.js_sub = self.create_subscription(JointState, '/joint_states', self._js_cb, 100)
        self.ik_cli = self.create_client(GetPositionIK, '/compute_ik')
        self.traj_pub = self.create_publisher(JointTrajectory, controller_topic, 10)

        # /ee_anchor 는 나중에 구독자가 생겨도 받도록 latched QoS
        latched_qos = QoSProfile(
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST
        )
        self.ee_anchor_pub = self.create_publisher(PoseStamped, anchor_topic, latched_qos)
        self.ee_pose_cmd_pub = self.create_publisher(PoseStamped, ee_cmd_topic, 10)

        self._target_pose = None
        self._ack = False
        self.echo_sub = self.create_subscription(
            PoseStamped, '/ee_anchor_echo', self._echo_cb, 10
        )
        self._anchor_timer = None

    def _echo_cb(self, msg: PoseStamped):
        self._ack = True
        self.get_logger().info('Received /ee_anchor_echo. Stop publishing /ee_anchor.')
        if self._anchor_timer is not None:
            self._anchor_timer.cancel()
            self._anchor_timer = None

    def _anchor_tick(self):
        if self._ack:
            return
        if self.ee_anchor_pub.get_subscription_count() == 0:
            return
        self.ee_anchor_pub.publish(self._target_pose)

    def _js_cb(self, msg: JointState):
        self.joint_state = msg

    def wait_for(self):
        self.get_logger().info('Waiting for /compute_ik ...')
        self.ik_cli.wait_for_service()
        self.get_logger().info('Waiting for /joint_states ...')
        while rclpy.ok() and self.joint_state is None:
            rclpy.spin_once(self, timeout_sec=0.1)
        self.get_logger().info('Ready.')

    def go_to(self, x, y, z, roll, pitch, yaw, move_time=3.0):
        target = PoseStamped()
        target.header.frame_id = self.base_frame
        target.header.stamp = self.get_clock().now().to_msg()
        target.pose.position.x = x
        target.pose.position.y = y
        target.pose.position.z = z
        qx, qy, qz, qw = rpy_to_quat(roll, pitch, yaw)
        target.pose.orientation.x = qx
        target.pose.orientation.y = qy
        target.pose.orientation.z = qz
        target.pose.orientation.w = qw

        req = GetPositionIK.Request()
        req.ik_request.group_name = self.group_name
        req.ik_request.ik_link_name = self.ee_link
        req.ik_request.robot_state.joint_state = self.joint_state
        req.ik_request.pose_stamped = target
        req.ik_request.timeout.sec = 1
        req.ik_request.avoid_collisions = False

        fut = self.ik_cli.call_async(req)
        rclpy.spin_until_future_complete(self, fut)
        if not fut.result():
            raise RuntimeError('IK call failed')
        res = fut.result()
        if res.error_code.val != 1:
            raise RuntimeError(f'IK failed, code={res.error_code.val}')

        name_to_pos = dict(zip(res.solution.joint_state.name,
                               res.solution.joint_state.position))
        positions = []
        for jn in self.controller_joint_order:
            if jn not in name_to_pos:
                raise RuntimeError(f'IK solution missing joint: {jn}')
            positions.append(float(name_to_pos[jn]))

        traj = JointTrajectory()
        traj.joint_names = self.controller_joint_order
        pt = JointTrajectoryPoint()
        pt.positions = positions
        pt.time_from_start.sec = int(move_time)
        pt.time_from_start.nanosec = int((move_time - int(move_time)) * 1e9)
        traj.points.append(pt)
        self.traj_pub.publish(traj)
        self.get_logger().info(f'Published JointTrajectory ({move_time:.2f}s)')

        # 1) 앵커/명령 1회 즉시 송출
        self.ee_anchor_pub.publish(target)  # latched
        self.ee_pose_cmd_pub.publish(target)  # info

        # 2) 반복 퍼블리시 시작(예: 5Hz)
        self._target_pose = target
        self._anchor_timer = self.create_timer(0.2, self._anchor_tick)

        # 3) ACK 올 때까지 대기
        while rclpy.ok() and not self._ack:
            rclpy.spin_once(self, timeout_sec=0.1)

        # 4) 중지
        self.get_logger().info('Anchor handshake done. Publishing stopped.')


# ------------------ 추가: 그리퍼 전송 유틸 ------------------
def _send_gripper_via_action(node: Node, action_name: str, pos: float, effort: float) -> bool:
    if not action_name:
        return False
    ac = ActionClient(node, GripperCommand, action_name)
    if not ac.wait_for_server(timeout_sec=1.0):
        node.get_logger().warn(f'[grip] action server not available: {action_name}')
        return False
    goal = GripperCommand.Goal()
    goal.command.position = float(pos)
    goal.command.max_effort = float(effort)
    send_fut = ac.send_goal_async(goal)
    rclpy.spin_until_future_complete(node, send_fut)
    res_fut = send_fut.result().get_result_async()
    rclpy.spin_until_future_complete(node, res_fut)
    node.get_logger().info(f'[grip] action sent: pos={pos:.3f}, effort={effort:.1f}')
    return True


def _send_gripper_via_traj(node: Node, topic: str, joint_name: str, pos: float, move_time: float):
    pub = node.create_publisher(JointTrajectory, topic, 10)
    traj = JointTrajectory()
    traj.joint_names = [joint_name]
    pt = JointTrajectoryPoint()
    pt.positions = [float(pos)]
    pt.time_from_start.sec = int(move_time)
    pt.time_from_start.nanosec = int((move_time - int(move_time)) * 1e9)
    traj.points.append(pt)
    pub.publish(traj)
    node.get_logger().info(
        f'[grip] traj published: joint={joint_name}, pos={pos:.3f}, t={move_time:.2f}s -> {topic}'
    )
# -----------------------------------------------------------


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--x', type=float, required=True)
    parser.add_argument('--y', type=float, required=True)
    parser.add_argument('--z', type=float, required=True)
    parser.add_argument('--roll', type=float, default=0.0)
    parser.add_argument('--pitch', type=float, default=0.0)
    parser.add_argument('--yaw', type=float, default=0.0)
    parser.add_argument('--group', default='arm')
    parser.add_argument('--ee', default='end_effector_link')  # gen3 lite 예시
    parser.add_argument('--controller', default='/joint_trajectory_controller/joint_trajectory')
    parser.add_argument('--base', default='base_link')
    parser.add_argument('--move_time', type=float, default=3.0)
    parser.add_argument('--anchor_topic', default='/ee_anchor')
    parser.add_argument('--ee_cmd_topic', default='/ee_pose_cmd')
    parser.add_argument('--j1', default='joint_1'); parser.add_argument('--j2', default='joint_2')
    parser.add_argument('--j3', default='joint_3'); parser.add_argument('--j4', default='joint_4')
    parser.add_argument('--j5', default='joint_5'); parser.add_argument('--j6', default='joint_6')

    # ★ 추가: 초기 그리퍼 관련 옵션
    parser.add_argument('--init_gripper', type=float, default=None,
                        help='초기 그리퍼 position (예: 0.20). 지정 안하면 실행 안함')
    parser.add_argument('--gripper_action', default='/gen3_lite_2f_gripper_controller/gripper_cmd',
                        help='GripperCommand 액션 서버 이름')
    parser.add_argument('--gripper_effort', type=float, default=30.0,
                        help='그리퍼 액션의 max_effort')
    parser.add_argument('--gripper_controller', default='/gripper_controller/joint_trajectory',
                        help='JointTrajectory로 보낼 컨트롤러 토픽')
    parser.add_argument('--gripper_joint', default='gripper',
                        help='그리퍼 조인트 이름')
    parser.add_argument('--grip_move_time', type=float, default=0.5,
                        help='그리퍼 트젝 time_from_start (sec)')

    args = parser.parse_args()
    joint_order = [args.j1, args.j2, args.j3, args.j4, args.j5, args.j6]

    rclpy.init()
    node = IKToTrajInitAnchor(args.group, args.ee, args.controller, args.base,
                              joint_order, args.anchor_topic, args.ee_cmd_topic)
    try:
        node.wait_for()

        # ★ 초기 그리퍼 세팅 (있을 때만)
        if args.init_gripper is not None:
            ok = _send_gripper_via_action(node, args.gripper_action, args.init_gripper, args.gripper_effort)
            if not ok:
                _send_gripper_via_traj(node, args.gripper_controller, args.gripper_joint,
                                       args.init_gripper, args.grip_move_time)
            # (선택) 팔 보내기 전에 안정화를 위해 잠깐 대기
            time.sleep(0.2)

        node.go_to(args.x, args.y, args.z, args.roll, args.pitch, args.yaw, args.move_time)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
