# 1. related work (with paper)
# 2. Contents for our works
## 2-1. environment
- Ubuntu 22.04
## 2-2. settings
- gen3 lite model
- realsense D435i camera
- two cups & two blocks
- HTC VIVE PRO: 1 controller & two base stations
## 2-3. scenario
- blocks 를 같은 색깔의 컵에 넣기 (pick-and-place)
## 2-4. data
- 수작업 (1000개)
## 2-5. training model
- resnet + mlp
# 3. Commands
```
#터미널 1) Bringup
export ROS_DOMAIN_ID=3
ros2 launch kortex_bringup gen3_lite.launch.py \
  robot_ip:=192.168.1.10 dof:=6 gripper:=gen3_lite_2f launch_rviz:=false


#터미널 2) Moveit move_group
export ROS_DOMAIN_ID=3
ros2 launch kinova_gen3_lite_moveit_config move_group.launch.py publish_robot_description:=true


#터미널 3) 자세제어
export ROS_DOMAIN_ID=3
rm -rf build install log
source /opt/ros/humble/setup.bash
cd ~/ws_kortex
colcon build --symlink-install --packages-select gen3lite_teleop_wo_servo
source install/setup.bash  

ros2 topic pub --once /joint_trajectory_controller/joint_trajectory trajectory_msgs/JointTrajectory "{
  header: {stamp: {sec: 0, nanosec: 0}},
  joint_names: ['joint_1','joint_2','joint_3','joint_4','joint_5','joint_6'],
  points: [
    {
      positions: [-0.07953051226525432, -0.14539474708065825, 2.0280866523079664, 1.5212194525741993, 0.932631556894697, -1.5922299798800719],
      time_from_start: {sec: 10, nanosec: 0}
    }
  ]
}"
  
ros2 run gen3lite_teleop_wo_servo ik_to_traj_2 \
  --x 0.247 --y -0.023 --z 0.274 \
  --roll -3.100 --pitch 0.032 --yaw 3.113 \
  --group arm --ee end_effector_link \
  --controller /joint_trajectory_controller/joint_trajectory \
  --base base_link --move_time 5.0 \
  --init_gripper 0.20 \
  --gripper_action /gen3_lite_2f_gripper_controller/gripper_cmd \
  --gripper_effort 30.0


# 터미널 4)steamvr
export ROS_DOMAIN_ID=3
source ./.venv/openvr/bin/activate
cd ~/hdd/Developments/robot_manipulation/openvr_test
source /opt/ros/humble/setup.bash
source ~/ws_kortex/install/setup.bash

python steamvr_node_2.py

  
#터미널 5) 
export ROS_DOMAIN_ID=3
rm -rf build install log
source /opt/ros/humble/setup.bash
cd ~/ws_kortex
colcon build --symlink-install --packages-select gen3lite_teleop_wo_servo
source install/setup.bash
  
ros2 run gen3lite_teleop_wo_servo vr_ik_follower_sub_2 -- \
  --vr_topic /steamvr/right \
  --joy_topic /steamvr/right/joy \
  --controller /joint_trajectory_controller/joint_trajectory \
  --group arm --ee end_effector_link --base base_link \
  --rate 10.0 --move_time 0.2 \
  --pos_only \
  --R_vr2base 1 0 0  0 0 -1  0 1 0

# 터미널 6
pkill -f realsense || true
export ROS_DOMAIN_ID=3
source /opt/ros/humble/setup.bash

ros2 launch realsense2_camera rs_launch.py \
  namespace:=/camera \
  camera_name:=d435 \
  enable_color:=true enable_depth:=true \
  rgb_camera.color_profile:=424x240x6 \
  depth_module.depth_profile:=424x240x6 \
  align_depth:=true enable_sync:=true \
  pointcloud.enable:=false \
  initial_reset:=true


# 새 터미널 7 (레코더)
export ROS_DOMAIN_ID=3
rm -rf build install log
source /opt/ros/humble/setup.bash
cd ~/ws_kortex
colcon build --symlink-install --packages-select gen3lite_data_recorder
source install/setup.bash

ros2 run gen3lite_data_recorder sync_recorder --ros-args \
  -p rgb_topic:=/camera/d435/color/image_raw \
  -p depth_topic:=/camera/d435/depth/image_rect_raw \
  -p camera_info_topic:=/camera/d435/color/camera_info \
  -p joint_topic:=/joint_states \
  -p base_frame:=base_link -p ee_frame:=end_effector_link \
  -p sync_slop_sec:=0.2 -p queue_size:=100 \
  -p gate_by_vr_active:=true \
  -p active_timeout_sec:=0.08 \
  -p output:=/home/robot/hdd/Developments/robot_manipulation/data/scenario_0001.npz
```

</br>

```
현재 ee 위치 받기
ros2 run tf2_ros tf2_echo base_link end_effector_link


cd ~/ws_kortex
export ROS_DOMAIN_ID=3
rm -rf build install log
colcon build --symlink-install --packages-select gen3lite_teleop_wo_servo
source install/setup.bash
  
ros2 run gen3lite_teleop_wo_servo vr_ik_follower_sub_2 -- \
  --vr_topic /steamvr/right \
  --joy_topic /steamvr/right/joy \
  --controller /joint_trajectory_controller/joint_trajectory \
  --group arm --ee end_effector_link --base base_link \
  --rate 10.0 --move_time 0.2 \
  --pos_only \
  --R_vr2base 1 0 0  0 0 1  0 1 0

- 저장 포맷(.npz) 예시
rgb: [N, H, W, 3], uint8 (RGB)
depth: [N, H, W], float32 (미터)
q: [N, J], float32 (Kinova joint positions1)
dq(있다면): [N, J]
ee_xyz: [N, 3] (base_link 기준 EE 위치)
ee_rpy: [N, 3] (roll, pitch, yaw, 라디안)
stamp_ns: [N], int64 (ROS 시간)
joint_names: 객체 배열
camera_K, camera_D, camera_width, camera_height (있을 경우)


- 카메라 rviz 확인
ros2 run realsense2_camera realsense2_camera_node
rviz2


export ROS_DOMAIN_ID=3
source /opt/ros/humble/setup.bash
rviz2


# 확인용
ros2 topic echo /camera/d435/color/camera_info --once
ros2 topic echo /camera/d435/depth/camera_info --once
# → width: 640, height: 480가 나와야 정상

지원 조합 확인
rs-enumerate-devices -c



```
