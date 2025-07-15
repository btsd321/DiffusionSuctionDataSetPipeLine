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
# 获取当前文件的绝对路径
FILE_PATH = os.path.abspath(__file__)
# 获取当前目录（apply_dataset）的路径
FILE_DIR_generate_dataset = os.path.dirname(FILE_PATH)
# 获取项目根目录路径
FILE_DIR = os.path.dirname(FILE_DIR_generate_dataset)

# 将项目根目录添加到Python路径中，以便导入其他模块
sys.path.append(FILE_DIR)
# 数据生成器的参数配置文件路径
DATASET_apply_dataset_parameter = os.path.join(FILE_DIR_generate_dataset, 'parameter.json')
    
# 导入H5数据生成器模块，包含所有数据处理的核心功能
from H5DataGenerator import *

# 定义H5数据集的根输出目录
OUT_ROOT_DIR = os.path.join(FILE_DIR, 'h5_dataset')
if not os.path.exists(OUT_ROOT_DIR):
    os.makedirs(OUT_ROOT_DIR)    

# 定义训练集的输出目录
TRAIN_SET_DIR = os.path.join(OUT_ROOT_DIR, 'train')
if not os.path.exists(TRAIN_SET_DIR):
    os.mkdir(TRAIN_SET_DIR)

# 定义各类输入数据的目录路径
GT_DIR = os.path.join(FILE_DIR, 'gt')  # 真值标签目录：物体6D位姿等ground truth数据
SEGMENT_DIR = os.path.join(FILE_DIR, 'segment_images')  # 分割图像目录：EXR格式，包含物体ID和分割掩码
DEPTH_DIR = os.path.join(FILE_DIR, 'depth_images')  # 深度图像目录：PNG格式，用于3D点云重建
OBJ_PATH = os.path.join(FILE_DIR, 'OBJ')  # 3D物体模型目录：包含OBJ格式的物体几何模型
GT_PATH = os.path.join(FILE_DIR, 'gt')  # 真值数据路径：CSV格式的物体位姿标注
INDIVIDUA_PATH = os.path.join(FILE_DIR, 'individual_object_size')  # 单个物体尺寸标签目录：物体可见面积比例数据

if __name__ == "__main__":
    # 设置要处理的数据范围
    # 循环编号范围：每个循环代表一组独立的场景数据
    CYCLE_idx_list = range(81, 100)  # 处理第81-99个循环（共19个循环）
    # 场景编号范围：每个循环内包含的场景数量
    SCENE_idx_list = range(1, 51)    # 处理1-50号场景（每个循环50个场景）

    # --------------------------------------------------------------------------
    # 调试模式：用于调试法线估计算法时，只处理少量数据以加快测试速度
    CYCLE_idx_list = range(0, 1)     # 仅处理第0个循环
    SCENE_idx_list = range(1, 51)    # 仍处理1-50号场景
    # --------------------------------------------------------------------------

    # 创建H5数据生成器实例，加载参数配置
    # 该实例负责将多模态原始数据转换为标准化的H5训练数据
    g = H5DataGenerator(DATASET_apply_dataset_parameter)
    
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
        
            # 2. 构建分割图像文件路径并加载
            # 分割图像包含每个像素对应的物体ID，用于物体识别和分离
            seg_img_path = os.path.join(SEGMENT_DIR, 'cycle_{:0>4}'.format(cycle_id), 
                                       "{:0>3}".format(scene_id), 'Image0001.exr')
            segment_image = cv2.imread(seg_img_path, cv2.IMREAD_UNCHANGED)

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

