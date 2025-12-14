# robot simulation

import time
import numpy as np
import pybullet as p
import pybullet_data
import os
import random
import cv2
import sys
from os.path import dirname, realpath

from pybullet_engine.client import BulletClient
from pybullet_engine.models.panda.panda2 import Panda
from pybullet_engine.utils import *

def main():
    urdfRootPath = pybullet_data.getDataPath()

    # 1) Connect to PyBullet simulator in GUI mode
    client = BulletClient(is_gui=True) # False

    # 2) Load the Panda robot
    panda = Panda(client=client, pos=(0, 0, 0), quat=(0, 0, 0, 1), use_magic_gripper=False)

    # 3) Create the floor, container, and objects
    # Load the plane (ground surface)
    plane_id = client.load_urdf("pybullet_data://plane.urdf", pos=(0, 0, 0), quat=(0, 0, 0, 1), static=True)

    # Create and load a container
    container_size = [0.5, 0.5, 0.04]
    container_pos = (0.5, 0, container_size[2] / 2.0)
    client.create_container_urdf(size=container_size, output_filename="container")
    # container_id = client.load_urdf("assets://container/container.urdf", pos=container_pos,
    #                                 quat=(0, 0, 0, 1), static=True)

    bowl = client.load_urdf("assets://green_bowl/model.urdf")
    theta_bowl = random.uniform(-np.pi, np.pi)
    bowl_euler = (0, 0, theta_bowl)
    bowl_quat = p.getQuaternionFromEuler(bowl_euler)
    objects = [(bowl, "bowl")]

    bowl_pos = [random.uniform(0.5, 0.7), random.uniform(-0.13, 0.13), 0.03]

    p.resetBasePositionAndOrientation(bowl, bowl_pos, bowl_quat)

    selected_object, object_name = objects[0]
    object_pos = p.getBasePositionAndOrientation(selected_object)[0]

    print(f"[INFO] Selected object: {object_name}")


    # ----------------------------------------------------------------
    # 4. 고정 카메라 설정 및 초기 화면 출력
    # ----------------------------------------------------------------
    # 카메라 회전: (roll=0, pitch=180, yaw=0) → 라디안 변환 후 quaternion 변환
    rpy_rad = np.radians((0, 180, 0))
    fixed_quat = p.getQuaternionFromEuler(rpy_rad)
    fixed_cam_pos = (0.5, 0.0, 0.6)  # 컨테이너 상단 근처

    # 초기 렌더링을 위해 한 스텝 실행
    client.step(1)

    # 10회 반복하며 카메라 화면 출력 ('q' 입력 시 종료)
    for _ in range(10):
        cont, color_img, depth_img, segm_img = show_fixed_camera(client, fixed_cam_pos, fixed_quat)
        if not cont:
            print("[WARN] user pressed 'q'.")
            break
        client.step(1)
        time.sleep(0.01)

    # Move the robot to its home position
    print("[INFO] Moving robot to home position...")
    panda.move_home(timeout=5.0)
    time.sleep(1.0)

    # 저장 데이터
    depth_data = []
    joint_data = []
    seq_len = []
    obj_transform = []

    tmp_joint = []

    # 초기 home 상태의 joint 및 초기 joint, obj_transform 저장
    joint_home = panda.get_joint_pos()
    tmp_joint.append(joint_home)
    bowl_euler = list(bowl_euler)
    tmp_obj = bowl_pos
    tmp_obj.extend(bowl_euler)

    depth_data.append(depth_img)
    obj_transform.append(tmp_obj)

    offset = 0.005
    radius = 0.063 - offset

    if object_name == 'bowl':
        pick_pos = (object_pos[0] + np.cos(theta_bowl) * radius,
                    object_pos[1] + np.sin(theta_bowl) * radius,
                    object_pos[2] + offset)

        #slope_angle = np.random.uniform(0, np.pi / 3)

        if theta_bowl >= 0:
            pick_euler = (np.pi / 1.07, 0, theta_bowl - np.pi / 2)
        elif theta_bowl < 0:
            pick_euler = (-np.pi / 1.07, 0, theta_bowl + np.pi / 2)
            
        pick_quat = p.getQuaternionFromEuler(pick_euler)
    
        grasp_pos = pick_pos
        grasp_quat = pick_quat
    
        place_pos = (0.45, 0.0, 0.5)
    
        place_quat = pick_quat

    # Calculate inverse kinematics (IK) for pick, grasp, and place poses
    print("[INFO] Calculating IK for pick pose...")
    pick_q = panda.ik_pybullet(pos=pick_pos, quat=pick_quat)
    if pick_q is None:
        print("[WARN] IK for pick pose failed.")
        # client.wait_for_user("Press enter to quit.")
        client.disconnect()
        return

    print("[INFO] Calculating IK for grasp pose...")
    grasp_q = panda.ik_pybullet(pos=grasp_pos, quat=grasp_quat)
    if grasp_q is None:
        print("[WARN] IK for grasp pose failed.")
        # client.wait_for_user("Press enter to quit.")
        client.disconnect()
        return

    print("[INFO] Calculating IK for place pose...")
    place_q = panda.ik_pybullet(pos=place_pos, quat=place_quat)
    if place_q is None:
        print("[WARN] IK for place pose failed.")
        # client.wait_for_user("Press enter to quit.")
        client.disconnect()
        return

    # Plan paths using RRT (Rapidly-exploring Random Tree)
    # Path: home -> pick position
    start_q = panda.get_qpos()
    print("[INFO] Searching RRT path: start->pick")
    success_pick, path_pick = panda.rrt_collision_free(pos1=start_q, pos2=pick_q,
                                                       smooth_fine_path=False, disable_renderer=False)
    if not success_pick or path_pick is None:
        print("[WARN] RRT start->pick failed.")
        # client.wait_for_user("Press enter to quit.")
        client.disconnect()
        return

    print(f"[INFO] Found path to pick. Length={len(path_pick)}")

    # Path: pick position -> grasp position
    print("[INFO] Searching RRT path: pick->grasp")
    success_grasp, path_grasp = panda.rrt_collision_free(pos1=pick_q, pos2=grasp_q,
                                                       smooth_fine_path=False, disable_renderer=False)
    if not success_grasp or path_grasp is None:
        print("[WARN] RRT pick->grasp failed.")
        # client.wait_for_user("Press enter to quit.")
        client.disconnect()
        return

    print(f"[INFO] Found path to grasp. Length={len(path_grasp)}")

    # Path: grasp position -> place position
    print("[INFO] Searching RRT path: grasp->place")
    success_place, path_place = panda.rrt_collision_free(pos1=grasp_q, pos2=place_q,
                                                         smooth_fine_path=False, disable_renderer=False)
    if not success_place or path_place is None:
        print("[WARN] RRT grasp->place failed.")
        # client.wait_for_user("Press enter to quit.")
        client.disconnect()
        return

    print(f"[INFO] Found path to place. Length={len(path_place)}")

    # Execute the planned paths
    print("[INFO] Moving along path_pick...")
    _, tmp_joint_list = panda.move_qpos_trajectory(path_pick, speed=0.01, timeout=15)
    tmp_joint.extend(tmp_joint_list)

    print("[INFO] Moving along path_grasp...")
    _, tmp_joint_list = panda.move_qpos_trajectory(path_grasp, speed=0.01, timeout=15)
    tmp_joint.extend(tmp_joint_list)

    print("[INFO] Closing gripper to grasp the object...")
    _, tmp_joint_list = panda.close_gripper_free(timeout=2.0)
    tmp_joint.extend(tmp_joint_list)  # 그리퍼 닫은 후의 상태 추가
    grasp_idx = len(tmp_joint)  # 그 후의 joint 상태를 기록

    print("[INFO] Moving along path_place...")
    _, tmp_joint_list = panda.move_qpos_trajectory(path_place, speed=0.01, timeout=15)
    tmp_joint.extend(tmp_joint_list)

    # 현재 tmp_joint 길이를 확인
    current_length = len(tmp_joint)
    seq_len.append(current_length)

    # 목표 길이
    target_length = 1500



    # 패딩 추가
    if current_length < target_length:
        padding_length = target_length - current_length
        # 패딩을 NumPy 배열로 생성
        padding = np.zeros((padding_length, 9))  # (padding_length, 9) 형태로 패딩
        tmp_joint.extend(padding.tolist())  # NumPy 배열을 리스트로 변환하여 추가
    else:
        client.disconnect()
        return

    tmp_joint = np.array(tmp_joint)
    joint_data.append(tmp_joint)

    # for i in range(current_length):
    #     print(joint_data[0][i].shape)

    state_object, _ = p.getBasePositionAndOrientation(bowl)

    if state_object[2] > 0.4:
        done = True
    else:
        done = False


    # Disconnect from the simulation
    # time.sleep(3.0)
    client.disconnect()

    # OpenCV 창 종료 및 클라이언트 연결 해제
    cv2.destroyAllWindows()
    client.disconnect()

    if seq_len is not None:
        return depth_data, joint_data, seq_len, obj_transform, done, object_name, grasp_idx
    else:
        return None, None, None, None, done, object_name

def load_existing_data(dir_data, order, done, object_name):
    try:
        depth_data = np.load(f'{dir_data}/{done}_{object_name}_depth_visuals_{order:03d}.npz')
        joint_data = np.load(f'{dir_data}/{done}_{object_name}_joint_poses_{order:03d}.npz')
        seq_data = np.load(f'{dir_data}/{done}_{object_name}_seq_len_{order:03d}.npz')
        obj_data = np.load(f'{dir_data}/{done}_{object_name}_obj_transform_{order:03d}.npz')
        grasp_data = np.load(f'{dir_data}/{done}_{object_name}_grasp_idx_{order:03d}.npz')

        existing_depth_data = depth_data['depth_visuals']
        existing_joint_data = joint_data['joint_poses']
        existing_seq_data = seq_data['seq_len']
        existing_obj_data = obj_data['obj_transform']
        existing_grasp_data = grasp_data['grasp_idx']

        del depth_data, joint_data, seq_data, obj_data, grasp_data
        return existing_depth_data, existing_obj_data, existing_joint_data, existing_seq_data, existing_grasp_data

    except FileNotFoundError:
        return None, None, None, None, None



def save_data(dir_data, depth_visuals, joint_data, seq_len, obj_transform, grasp_idx,
              successful_order, fail_order, done, object_name):
    max_save_len = 1000

    if done:
        done_str = "success"
        order = successful_order
    else:
        done_str = "fail"
        order = fail_order

    # 파일이 존재하면 불러오고 없으면 새로운 데이터를 생성
    while True:
        if len(joint_data) > 0:
            existing_depth_data, existing_obj_data, existing_joint_data, existing_seq_data, existing_grasp_data = \
                load_existing_data(dir_data, order, done_str, object_name)

            # 데이터가 없다면 새로운 데이터로 초기화
            if existing_joint_data is None and order == 0:
                new_depth_data = np.array(depth_visuals)
                new_joint_data = np.array(joint_data)
                new_seq_data = np.array(seq_len)
                new_obj_data = np.array(obj_transform)
                new_grasp_data = np.array(grasp_idx)
            elif existing_joint_data is None:
                new_depth_data = np.array(depth_visuals)
                new_joint_data = np.array(joint_data)
                new_seq_data = np.array(seq_len)
                new_obj_data = np.array(obj_transform)
                new_grasp_data = np.array(grasp_idx)
            elif existing_joint_data.shape[0] > max_save_len:
                if done_str == "success":
                    successful_order += 1
                    order = successful_order
                    continue
                else:
                    fail_order += 1
                    order = fail_order
                    continue
            else:
                # 데이터를 합쳐서 저장
                new_depth_data = np.vstack((existing_depth_data, np.array(depth_visuals)))
                new_joint_data = np.vstack((existing_joint_data, np.array(joint_data)))
                new_seq_data = np.vstack((existing_seq_data, np.array(seq_len)))
                new_obj_data = np.vstack((existing_obj_data, np.array(obj_transform)))
                new_grasp_data = np.vstack((existing_grasp_data, np.array(grasp_idx)))

                del existing_depth_data, existing_obj_data, existing_joint_data, existing_seq_data, existing_grasp_data

            # 저장 경로 설정 및 저장
            save_path_depth = f'{dir_data}/{done_str}_{object_name}_depth_visuals_{order:03d}.npz'
            save_path_joint = f'{dir_data}/{done_str}_{object_name}_joint_poses_{order:03d}.npz'
            save_path_seq_len = f'{dir_data}/{done_str}_{object_name}_seq_len_{order:03d}.npz'
            save_path_obj = f'{dir_data}/{done_str}_{object_name}_obj_transform_{order:03d}.npz'
            save_path_grasp = f'{dir_data}/{done_str}_{object_name}_grasp_idx_{order:03d}.npz'

            np.savez(save_path_depth, depth_visuals=new_depth_data)
            np.savez(save_path_joint, joint_poses=new_joint_data)
            np.savez(save_path_seq_len, seq_len=new_seq_data)
            np.savez(save_path_obj, obj_transform=new_obj_data)
            np.savez(save_path_grasp, grasp_idx=new_grasp_data)

            print('intermediate save')
            return successful_order, fail_order
        else:
            return successful_order, fail_order


if __name__ == "__main__":
    dir_project = dirname(dirname(realpath(__file__)))  # Get parent-dir name
    sys.path.append(dir_project)

    dir_data = f"{dir_project}/sim_manipulation_vae/dataset2"
    successful_order = 0
    fail_order = 0
    for _ in range(10):
        for _ in range(500):
            result = main()
            if result is not None:
                # 기존: depth_data, joint_data, seq_len, obj_transform, done, object_name = result
                depth_data, joint_data, seq_len, obj_transform, done, object_name, grasp_idx = result
                successful_order, fail_order = save_data(dir_data, depth_data, joint_data, seq_len,
                                                         obj_transform, grasp_idx,
                                                         successful_order, fail_order, done, object_name)
            else:
                print("[INFO] main() returned None, skipping this iteration.")
