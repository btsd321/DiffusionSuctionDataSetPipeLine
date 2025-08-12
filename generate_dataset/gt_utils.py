"""
本文件用于批量读取物理仿真结果csv, 计算物体在世界坐标系下的位姿, 并生成GT(Ground Truth)标注文件。
支持多线程并行处理，适用于大规模数据集标注自动生成流程。

主要功能:
- 并行处理多个循环和场景的GT生成
- 支持自定义线程数量
- 详细的进度显示和错误处理
- 直接返回世界坐标系下的位姿参数

作者: Huang Dingtao
校验: Huang Dingtao
更新: 添加多线程支持
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
parser.add_argument('--max_workers', type=int, default=4, help='线程池最大线程数')
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
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

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
    直接返回世界坐标系下的位姿
    输入参数:
        pose_world: 零件在世界坐标系下的位姿 (N, 7) [x, y, z, qx, qy, qz, qw]
    返回:
        pose_world_matrix: 零件在世界坐标系下的位姿 (N, 12) [x, y, z, R1~R9]
    '''  
    
    # 直接使用世界坐标系下的平移和旋转，无需坐标系转换
    t_world = pose_world[:,:3]                       # 物体在世界坐标系下的平移
    quat_world = pose_world[:,3:]                    # 物体在世界坐标系下的旋转(四元数)
    R_world = np.array([nq.quat2mat(quat).reshape(9) for quat in quat_world])  # 转换为旋转矩阵并展平
    
    # 拼接平移和旋转为最终结果
    pose_world_matrix = np.concatenate((t_world, R_world), axis=-1)
    return pose_world_matrix

def process_single_scene(cycle_id, scene_id):
    """
    处理单个场景的GT生成
    
    参数:
        cycle_id: 循环编号
        scene_id: 场景编号
        
    返回:
        tuple: (cycle_id, scene_id, success, error_msg)
    """
    try:
        # 构建当前循环和场景的csv路径
        csv_path = os.path.join(OUTDIR_physics_result_dir, 'cycle_{:0>4}'.format(cycle_id), 
                               "{:0>3}".format(scene_id), "{:0>3}.csv".format(scene_id))
        
        # 检查输入文件是否存在
        if not os.path.exists(csv_path):
            return (cycle_id, scene_id, False, f"输入文件不存在: {csv_path}")
        
        # 读取数据
        name_temp, index, pose_world = read_csv(csv_path)
        
        # 生成世界坐标系下的位姿
        pose_world_matrix = generate_gt(pose_world)
        
        # 构建csv表头
        headers = ["class_name", "id", "x", "y", "z", "R1", "R2", "R3", "R4", "R5", "R6", "R7", "R8", "R9"]
        
        # 拼接物体名称、索引和位姿
        temp = np.concatenate((name_temp.reshape(-1, 1), index.reshape(-1, 1)), axis=-1)
        temp = np.concatenate((temp, pose_world_matrix), axis=-1)
        result = temp.tolist()
        
        assert len(result[0]) == len(headers)
        
        # 构建保存路径
        save_path = os.path.join(GT_PATH, 'cycle_{:0>4}'.format(cycle_id), "{:0>3}".format(scene_id))
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        file_loc = os.path.join(save_path, '{:0>3}.csv'.format(scene_id))
        
        # 写入csv文件
        with open(file_loc, 'w', newline='') as f:
            f_csv = csv.writer(f)
            f_csv.writerow(headers)
            f_csv.writerows(result)
        
        return (cycle_id, scene_id, True, f"成功处理场景 cycle_{cycle_id:04d}/scene_{scene_id:03d}")
        
    except Exception as e:
        error_msg = f"处理场景 cycle_{cycle_id:04d}/scene_{scene_id:03d} 时发生错误: {str(e)}"
        return (cycle_id, scene_id, False, error_msg)

if __name__ == "__main__":
    print(f"开始批量生成GT，使用 {FLAGS.max_workers} 个线程")
    print(f"循环范围: {CYCLE_idx_list}")
    print(f"场景范围: {SCENE_idx_list}")
    
    # 创建所有需要处理的任务列表
    tasks = []
    for cycle_id in CYCLE_idx_list:
        for scene_id in SCENE_idx_list:
            tasks.append((cycle_id, scene_id))
    
    print(f"总共需要处理 {len(tasks)} 个场景")
    
    # 统计变量
    success_count = 0
    error_count = 0
    completed_cycles = set()
    
    # 使用线程池执行任务
    with ThreadPoolExecutor(max_workers=FLAGS.max_workers) as executor:
        # 提交所有任务
        future_to_task = {executor.submit(process_single_scene, cycle_id, scene_id): (cycle_id, scene_id) 
                         for cycle_id, scene_id in tasks}
        
        # 处理完成的任务
        for future in as_completed(future_to_task):
            cycle_id, scene_id, success, message = future.result()
            
            if success:
                success_count += 1
                completed_cycles.add(cycle_id)
                print(f"✓ {message}")
            else:
                error_count += 1
                print(f"✗ {message}")
            
            # 显示进度
            total_processed = success_count + error_count
            progress = (total_processed / len(tasks)) * 100
            print(f"进度: {total_processed}/{len(tasks)} ({progress:.1f}%)")
    
    # 输出完成统计
    print("\n" + "="*50)
    print("批量处理完成!")
    print(f"成功处理: {success_count} 个场景")
    print(f"失败处理: {error_count} 个场景")
    print(f"完成的循环: {sorted(completed_cycles)}")
    
    if error_count == 0:
        print("🎉 全部场景处理成功!")
    else:
        print(f"⚠️  有 {error_count} 个场景处理失败，请检查错误信息")
