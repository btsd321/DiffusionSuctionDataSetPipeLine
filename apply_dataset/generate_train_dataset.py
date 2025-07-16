"""
本文件用于生成机器人吸取任务的训练数据集，将多模态数据转换为H5格式。

主要功能：
1. 整合多种数据源：深度图像、分割图像、物体位姿真值、物体尺寸标签
2. 通过H5DataGenerator处理原始数据，生成统一格式的训练数据集
3. 支持批量处理多个循环(cycle)和场景(scene)的数据
4. 生成的H5数据集包含点云、法向量、各种吸取评分等信息

数据流程：
输入数据：
- 深度图像(depth_images): PNG格式，用于3D点云重建
- 分割图像(segment_images): EXR格式，包含物体分割信息和ID
- 物体位姿真值(gt): CSV格式，包含物体在相机坐标系下的6D位姿
- 物体尺寸标签(individual_object_size): CSV格式，物体可见面积比例信息

输出数据：
- H5数据集文件：包含点云坐标、法向量、密封评分、抗扭评分、可行性评分等

应用场景：
- 机器人抓取和吸取任务的深度学习训练
- 多模态感知数据的预处理和标准化
- 大规模仿真数据集的生成和管理

@author: Huang Dingtao
@checked: Huang Dingtao
"""
import os
# 设置CUDA可见设备为GPU 0，用于加速数据处理中的深度学习计算
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
# import torch
# test = torch.randn(30, 3).cuda()  # GPU测试代码（已注释）

import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'config'))
from camera_info import CameraInfo
import argparse
import re
import OpenEXR
import Imath
import numpy as np
    
# 导入H5数据生成器模块，包含所有数据处理的核心功能
from H5DataGenerator import *

# 命令行参数解析
parser = argparse.ArgumentParser()
# 数据集根目录
parser.add_argument('--data_dir', type=str, default='G:/Diffusion_Suction_DataSet', help='数据集根目录')
parser.add_argument('--cycle_list', type=str, required=True, help='循环编号，支持格式: "5"(单个), "[1,10]"(区间), "{1,3,5}"(列表)')
parser.add_argument('--scene_list', type=str, required=True, help='场景编号，支持格式: "5"(单个), "[1,10]"(区间), "{1,3,5}"(列表)')
parser.add_argument('--camera_info_file', type=str, default='camera_info.yaml', help='相机参数配置文件路径')
parser.add_argument('--parameter_file', type=str, default='parameter.json', help='数据生成器参数配置文件路径')
FLAGS = parser.parse_args()

def parse_range_or_single(input_str):
    """
    解析输入字符串，支持以下格式：
    - 单个值: "5" -> [5]
    - 区间: "[1,10]" -> [1,2,3,4,5,6,7,8,9,10]
    - 列表: "{1,3,5}" -> [1,3,5]
    """
    input_str = input_str.strip()
    
    # 如果是区间格式 [start,end]
    range_match = re.match(r'^\[(\d+),(\d+)\]$', input_str)
    if range_match:
        start, end = map(int, range_match.groups())
        return list(range(start, end + 1))
    
    # 如果是列表格式 {1,3,5,7}
    list_match = re.match(r'^\{(.+)\}$', input_str)
    if list_match:
        values_str = list_match.group(1)
        return [int(x.strip()) for x in values_str.split(',')]
    
    # 如果是单个数字
    if input_str.isdigit():
        return [int(input_str)]
    
    # 如果都不匹配，抛出错误
    raise ValueError(f"无法解析输入格式: {input_str}. 支持的格式: '5'(单个), '[1,10]'(区间), '{{1,3,5}}'(列表)")

# 定义H5数据集的根输出目录
OUT_ROOT_DIR = os.path.join(FLAGS.data_dir, 'h5_dataset')
if not os.path.exists(OUT_ROOT_DIR):
    os.makedirs(OUT_ROOT_DIR)    

# 定义训练集的输出目录
TRAIN_SET_DIR = os.path.join(FLAGS.data_dir, 'train')
if not os.path.exists(TRAIN_SET_DIR):
    os.mkdir(TRAIN_SET_DIR)

# 定义各类输入数据的目录路径
GT_DIR = os.path.join(FLAGS.data_dir, 'gt')  # 真值标签目录：物体6D位姿等ground truth数据
SEGMENT_DIR = os.path.join(FLAGS.data_dir, 'segment_images')  # 分割图像目录：EXR格式，包含物体ID和分割掩码
DEPTH_DIR = os.path.join(FLAGS.data_dir, 'depth_images')  # 深度图像目录：PNG格式，用于3D点云重建
OBJ_PATH = os.path.join(FLAGS.data_dir, 'OBJ')  # 3D物体模型目录：包含OBJ格式的物体几何模型
GT_PATH = os.path.join(FLAGS.data_dir, 'gt')  # 真值数据路径：CSV格式的物体位姿标注
INDIVIDUA_PATH = os.path.join(FLAGS.data_dir, 'individual_object_size')  # 单个物体尺寸标签目录：物体可见面积比例数据

# 用OpenEXR读取EXR分割图像
def read_exr_to_numpy(filepath):
    exr_file = OpenEXR.InputFile(filepath)
    header = exr_file.header()
    dw = header['dataWindow']
    width = dw.max.x - dw.min.x + 1
    height = dw.max.y - dw.min.y + 1
    channels = ['R', 'G', 'B']
    pt = Imath.PixelType(Imath.PixelType.FLOAT)
    data = [np.frombuffer(exr_file.channel(c, pt), dtype=np.float32) for c in channels]
    img = np.stack([d.reshape(height, width) for d in data], axis=-1)
    return img

if __name__ == "__main__":
    # 解析循环和场景编号为整数列表
    CYCLE_idx_list = parse_range_or_single(FLAGS.cycle_list)
    SCENE_idx_list = parse_range_or_single(FLAGS.scene_list)

    # 创建H5数据生成器实例，加载参数配置
    # 该实例负责将多模态原始数据转换为标准化的H5训练数据
    g = H5DataGenerator(params_file_name = FLAGS.parameter_file, 
                        camera_info_file_name = FLAGS.camera_info_file, 
                        objs_path = os.path.join(FLAGS.data_dir, 'OBJ'),
                        target_num_point=16384)
    
    # 外层循环：遍历所有指定的循环（数据批次）
    for cycle_id in CYCLE_idx_list:
        # 为当前循环创建输出目录
        out_cycle_dir = os.path.join(TRAIN_SET_DIR, 'cycle_{:0>4}'.format(cycle_id))
        if not os.path.exists(out_cycle_dir):
            os.mkdir(out_cycle_dir)
      
        # 内层循环：遍历当前循环下的所有场景
        for scene_id in SCENE_idx_list:
            # 1. 构建深度图像文件路径并加载
            # 深度图像用于重建3D点云，提供空间几何信息
            depth_image_path = os.path.join(DEPTH_DIR, 'cycle_{:0>4}'.format(cycle_id), 
                                          "{:0>3}".format(scene_id), 'Image0001.png')
            depth_image = cv2.imread(depth_image_path, cv2.IMREAD_UNCHANGED)
            if depth_image is None:
                raise ValueError(f"无法读取深度图像文件: {depth_image_path}")
        
            # 2. 构建分割图像文件路径并加载
            # 分割图像包含每个像素对应的物体ID，用于物体识别和分离
            seg_img_path = os.path.join(SEGMENT_DIR, 'cycle_{:0>4}'.format(cycle_id), 
                                       "{:0>3}".format(scene_id), 'Image0001.exr')

            segment_image = read_exr_to_numpy(seg_img_path)

            # 3. 构建真值标签文件路径
            # GT文件包含每个物体在相机坐标系下的6D位姿（位置+旋转）
            gt_file_path = os.path.join(GT_PATH, 'cycle_{:0>4}'.format(cycle_id), 
                                       "{:0>3}".format(scene_id), '{:0>3}.csv'.format(scene_id))

            # 4. 构建输出H5文件路径
            # H5文件将包含处理后的训练数据：点云、法向量、各种评分等
            output_h5_path = os.path.join(out_cycle_dir, "{:0>3}.h5".format(scene_id))

            # 5. 可选数据路径（当前已注释）
            # 稠密点云数据：用于更精细的几何分析（如果需要）
            # dense_point_path = os.path.join(OBJ_PATH + '_{}'.format(obj_id), '_{}'.format(obj_id)+".ply")

            # 单个物体的预计算抓取分数：用于KNN算法计算密封分数（如果需要）
            # seal_path = os.path.join(FILE_DIR, 'OBJ' + '_{}'.format(obj_id), '_{}'.format(obj_id) + ".npz")

            # 6. 构建单个物体尺寸标签文件路径
            # 该文件包含每个物体的可见面积比例，用于评估吸取可行性
            individual_object_size_path = os.path.join(INDIVIDUA_PATH, 'cycle_{:0>4}'.format(cycle_id), 
                                                     "{:0>3}".format(scene_id), '{:0>3}.csv'.format(scene_id))

            # 7. 核心处理步骤：调用数据生成器处理当前场景
            # 该函数将整合所有输入数据，生成标准化的H5训练数据集
            # 包括：点云生成、法向量估计、吸取评分计算、数据标准化等
            g.process_train_set(depth_image, segment_image, gt_file_path, 
                              output_h5_path, individual_object_size_path)
            
            # 打印处理进度
            print(f"已完成处理：循环 {cycle_id:04d}，场景 {scene_id:03d}")
    
    print("所有训练数据集生成完成！")

