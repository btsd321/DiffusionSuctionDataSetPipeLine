"""
本文件用于批量读取物理仿真结果csv, 计算物体在相机坐标系下的位姿, 并生成GT(Ground Truth)标注文件, 适用于数据集标注自动生成流程。

作者: Huang Dingtao
校验: Huang Dingtao
"""

import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'config'))
from camera_info import CameraInfo
import argparse

def parse_range_or_single(input_str):
    """
    解析输入字符串，支持以下格式：
    - 单个值: "5" -> [5]
    - 区间: "[1,10]" -> [1,2,3,4,5,6,7,8,9,10]
    - 列表: "{1,3,5}" -> [1,3,5]
    """
    input_str = input_str.strip()
    range_match = re.match(r'^\[(\d+),(\d+)\]$', input_str)
    if range_match:
        start, end = map(int, range_match.groups())
        return list(range(start, end + 1))
    list_match = re.match(r'^\{(.+)\}$', input_str)
    if list_match:
        values_str = list_match.group(1)
        return [int(x.strip()) for x in values_str.split(',')]
    if input_str.isdigit():
        return [int(input_str)]
    raise ValueError(f"无法解析输入格式: {input_str}. 支持的格式: '5'(单个), '[1,10]'(区间), '{{1,3,5}}'(列表)")

# 命令行参数解析
parser = argparse.ArgumentParser()
# 数据集根目录
parser.add_argument('--data_dir', type=str, default='G:/Diffusion_Suction_DataSet', help='数据集根目录')
# 循环编号
parser.add_argument('--cycle_list', type=str, required=True, 
                   help='循环编号，支持格式: "5"(单个), "[1,10]"(区间), "{1,3,5}"(列表)')
# 场景编号  
parser.add_argument('--scene_list', type=str, required=True, 
                   help='场景编号，支持格式: "5"(单个), "[1,10]"(区间), "{1,3,5}"(列表)')
parser.add_argument('--camera_info_file', type=str, default='camera_info.yaml', help='相机参数配置文件路径')
FLAGS = parser.parse_args()

# 获取数据集根目录
FILE_DIR = FLAGS.data_dir

import csv
import cv2
import math
import numpy as np
import os
import json
import nibabel.quaternions as nq
import yaml
import re

# 解析循环编号和场景编号
try:
    CYCLE_idx_list = parse_range_or_single(FLAGS.cycle_list)
    SCENE_idx_list = parse_range_or_single(FLAGS.scene_list)
except ValueError as e:
    print(f"参数解析错误: {e}")
    sys.exit(1)

# 加载相机参数
camera_info_path = FLAGS.camera_info_file
if not os.path.isabs(camera_info_path):
    camera_info_path = os.path.join(os.path.dirname(__file__), '..', 'config', camera_info_path)
camera_info_path = os.path.abspath(camera_info_path)
print(f"加载相机参数文件: {camera_info_path}")
cam_info = CameraInfo(camera_info_path)

# 从配置文件获取相机的空间位置和旋转(四元数)
CAMERA_LOCATION = cam_info.cam_translation_vector
CAMERA_ROTATION = cam_info.cam_quaternions

# 物理仿真结果和GT保存路径
OUTDIR_physics_result_dir =  os.path.join(FILE_DIR, 'physics_result')
GT_PATH =  os.path.join(FILE_DIR, 'gt')
if not os.path.exists(GT_PATH):
    os.makedirs(GT_PATH)    

def read_csv(csv_path):        
    """
    读取单个csv文件, 返回物体名称、索引和位姿
    输入:
        csv_path: csv文件路径
    输出:
        obj_name: 物体名称数组
        obj_index: 物体索引数组
        pose: 物体位姿数组(x, y, z, qx, qy, qz, qw)
    """
    with open(csv_path,'r') as csv_file:  
        all_lines=csv.reader(csv_file) 
        list_file = [i for i in all_lines]  
    array_file = np.array(list_file)[1:] # 去除表头
    obj_name = array_file[:,0]
    obj_index = array_file[:,1].astype('int')
    pose = array_file[:,2:9].astype('float32')
    return obj_name, obj_index, pose

def generate_gt(pose_world):
    ''' 
    生成相机坐标系下的位姿
    输入参数:
        pose_world: 零件在世界坐标系下的位姿 (N, 7) [x, y, z, qx, qy, qz, qw]
    返回:
        pose_camera: 零件在相机坐标系下的位姿 (N, 12) [x, y, z, R1~R9]
    '''  

    # 方式1: 使用外参矩阵进行变换（推荐）
    if cam_info.extrinsic_matrix is not None:
        # 从外参矩阵中提取旋转和平移
        extrinsic = cam_info.extrinsic_matrix  # 4x4外参矩阵
        R_w2c = extrinsic[:3, :3]  # 世界到相机的旋转矩阵
        t_w2c = extrinsic[:3, 3]   # 世界到相机的平移向量
        
        # 世界坐标系下的平移和旋转
        t_world = pose_world[:,:3]                       # 物体在世界坐标系下的平移
        quat_world = pose_world[:,3:]                    # 物体在世界坐标系下的旋转(四元数)
        R_world = [nq.quat2mat(quat) for quat in quat_world]  # 转换为旋转矩阵

        # 生成相机坐标系下的平移和旋转
        t_camera = np.array([np.dot(R_w2c, t) + t_w2c for t in t_world])  # 物体在相机坐标系下的平移
        R_camera = np.array([np.dot(R_w2c, R.reshape(3,3)).reshape(9) for R in R_world])  # 物体在相机坐标系下的旋转
        
    else:
        # 方式2: 使用四元数和平移向量（备选方案）
        print("警告: 外参矩阵不可用，使用四元数和平移向量进行变换")
        # 相机外参
        t_c2w = np.array(CAMERA_LOCATION).reshape(1, 3)  # 相机在世界坐标系下的平移
        quat_c2w = np.array(CAMERA_ROTATION)             # 相机在世界坐标系下的旋转(四元数)
        R_c2w  = nq.quat2mat(quat_c2w).reshape(3,3)      # 转换为旋转矩阵

        # 世界坐标系下的平移和旋转
        t_world = pose_world[:,:3]                       # 物体在世界坐标系下的平移
        quat_world = pose_world[:,3:]                    # 物体在世界坐标系下的旋转(四元数)
        R_world = [nq.quat2mat(quat) for quat in quat_world]  # 转换为旋转矩阵

        # 生成相机坐标系下的平移和旋转
        t_camera = np.array([(np.dot(R_c2w, t) + t_c2w).reshape(3) \
                          for t in t_world])            # 物体在相机坐标系下的平移
        R_camera = np.array([np.dot(R_c2w, R.reshape(3,3)).reshape(9) \
                          for R in R_world])            # 物体在相机坐标系下的旋转(展平成9维)
    
    pose_camera = np.concatenate((t_camera, R_camera),axis=-1)  # 拼接为最终结果
    return pose_camera

if __name__ == "__main__":
    # 遍历所有循环和场景, 批量生成GT
    for cycle_id in CYCLE_idx_list:
        for scene_id in SCENE_idx_list:
            # 构建当前循环和场景的csv路径
            csv_path = os.path.join(OUTDIR_physics_result_dir, 'cycle_{:0>4}'.format(cycle_id),"{:0>3}".format(scene_id), "{:0>3}.csv".format(scene_id))              
            name_temp, index, pose_world = read_csv(csv_path)
            
            # 生成相机坐标系下的位姿
            pose_camera = generate_gt(pose_world)
            # 构建csv表头
            headers = ["class_name","id","x", "y", "z", "R1", "R2", "R3", "R4","R5", "R6", "R7", "R8","R9"]
            # 拼接物体名称、索引和位姿
            temp = np.concatenate((name_temp.reshape(-1,1), index.reshape(-1,1)),axis=-1)
            temp = np.concatenate((temp, pose_camera),axis=-1)
            result = temp.tolist()
            
            assert len(result[0]) == len(headers)
            # 构建保存路径
            save_path = os.path.join(GT_PATH,'cycle_{:0>4}'.format(cycle_id),"{:0>3}".format(scene_id))
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            file_loc = save_path + '/' + '{:0>3}'.format(scene_id) + '.csv'
            
            # 写入csv文件
            with open(file_loc, 'w', newline='') as f:
                    f_csv = csv.writer(f)
                    f_csv.writerow(headers)
                    f_csv.writerows(result)
        print(f'第 {cycle_id} 个循环已完成')
    print('全部完成')
