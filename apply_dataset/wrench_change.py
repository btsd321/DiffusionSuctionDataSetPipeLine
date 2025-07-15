"""
本文件用于对已生成的H5数据集进行吸取抗扭分数的后处理调整。

主要功能：
1. 读取原始的H5数据集文件，包含点云数据和各种吸取评分
2. 根据吸取点法向量与参考方向(0,0,-1)的夹角，对吸取抗扭分数进行加权调整
3. 夹角越小（法向量越接近垂直向下），抗扭分数权重越高
4. 将调整后的数据保存到新的H5数据集文件中

应用场景：
- 优化机器人吸取任务中的抗扭转能力评估
- 考虑重力因素，优先选择垂直向下的吸取姿态
- 提高吸取成功率和稳定性

输入：原始H5数据集（h5_dataset目录）
输出：调整后的H5数据集（h5_dataset_change目录）

@author: Huang Dingtao
@checked: Huang Dingtao
"""
import os
# 设置CUDA可见设备为GPU 0
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
# import torch
# test = torch.randn(30, 3).cuda()  # GPU测试代码（已注释）

import sys
# 获取当前文件的绝对路径
FILE_PATH = os.path.abspath(__file__)
# 获取generate_dataset目录路径
FILE_DIR_generate_dataset = os.path.dirname(FILE_PATH)
# 获取项目根目录路径
FILE_DIR = os.path.dirname(FILE_DIR_generate_dataset)

# 将项目根目录添加到Python路径中，以便导入其他模块
sys.path.append(FILE_DIR)
# 参数配置文件路径
DATASET_apply_dataset_parameter = os.path.join(FILE_DIR_generate_dataset, 'parameter.json')
    
# 导入H5数据生成器模块
from H5DataGenerator import *

# 定义原始H5数据集路径
OUT_ROOT_DIR =  os.path.join(FILE_DIR, 'h5_dataset')
TRAIN_SET_DIR = os.path.join(OUT_ROOT_DIR, 'train')

# 定义调整后H5数据集的输出路径
OUT_ROOT_DIR_Change =  os.path.join(FILE_DIR, 'h5_dataset_change')
if not os.path.exists(OUT_ROOT_DIR_Change):
    os.makedirs(OUT_ROOT_DIR_Change)    
TRAIN_SET_DIR_Change = os.path.join(OUT_ROOT_DIR_Change, 'train')
if not os.path.exists( TRAIN_SET_DIR_Change ):
    os.mkdir(TRAIN_SET_DIR_Change)

def checke_angles_degrees(normals, score):
    """
    根据法向量与参考向量(0,0,-1)之间的夹角, 对分数进行加权调整。
    
    该函数实现了基于重力方向的吸取姿态优化策略：
    - 法向量越接近垂直向下(0,0,-1)，权重越高
    - 通过夹角计算，将角度映射到[0,1]的权重范围
    - 0度夹角权重为1.0，180度夹角权重为0.0

    参数:
        normals (numpy.ndarray): 法向量数组, 形状为(N, 3)，表示N个吸取点的法向量
        score (numpy.ndarray): 原始分数数组，形状为(N,)，表示对应的吸取抗扭分数

    返回:
        score (numpy.ndarray): 加权调整后的分数数组，形状为(N,)
    
    数学原理:
        1. 计算法向量与参考向量(0,0,-1)的点积
        2. 通过点积和向量模长计算夹角余弦值
        3. 将夹角转换为权重: weight = 1 - (angle_degrees / 180)
        4. 用权重调整原始分数: adjusted_score = original_score * weight
    """

    # 定义参考向量，指向负Z轴方向（垂直向下，符合重力方向）
    reference_vector = np.array([0, 0, -1])

    # 计算法向量与参考向量的点积（向量内积）
    dot_products = np.sum(normals * reference_vector, axis=1)

    # 计算每个法向量的模长（向量长度）
    norm_magnitudes = np.linalg.norm(normals, axis=1)
    # 计算参考向量的模长
    reference_magnitude = np.linalg.norm(reference_vector)

    # 根据向量点积公式计算夹角的余弦值: cos(θ) = (a·b) / (|a|*|b|)
    cos_angles = dot_products / (norm_magnitudes * reference_magnitude)

    # 防止数值计算误差导致余弦值超出[-1, 1]范围
    cos_angles = np.clip(cos_angles, -1.0, 1.0)

    # 通过反余弦函数计算夹角（弧度制）
    angles = np.arccos(cos_angles)

    # 将弧度转换为角度制，便于理解和调试
    angles_degrees = np.degrees(angles)

    # 将角度映射到[0,1]权重区间：
    # - 0度（完全垂直向下）→ 权重1.0
    # - 90度（水平方向）→ 权重0.5  
    # - 180度（完全垂直向上）→ 权重0.0
    mapped_values = 1 - (angles_degrees / 180)

    # 用计算得到的权重对原始分数进行加权调整
    score *= mapped_values
    
    return score

if __name__ == "__main__":
    # 定义要处理的数据范围
    CYCLE_idx_list = range(0, 1)    # 循环编号：处理第0个循环
    SCENE_idx_list = range(1, 51)   # 场景编号：处理1-50号场景

    # 创建H5数据生成器实例，用于数据处理
    g = H5DataGenerator(DATASET_apply_dataset_parameter)
    
    # 遍历所有指定的循环
    for cycle_id in CYCLE_idx_list:
        # 构建原始数据和输出数据的循环目录路径
        out_cycle_dir = os.path.join(TRAIN_SET_DIR, 'cycle_{:0>4}'.format(cycle_id))
        out_cycle_dir_change = os.path.join(TRAIN_SET_DIR_Change, 'cycle_{:0>4}'.format(cycle_id))
        
        # 创建输出目录（如果不存在）
        if not os.path.exists(out_cycle_dir_change):
            os.mkdir(out_cycle_dir_change)
            
        # 遍历当前循环下的所有场景
        for scene_id in SCENE_idx_list:
            # 定义原始H5文件和调整后H5文件的完整路径
            output_h5_path = os.path.join(out_cycle_dir, "{:0>3}.h5".format(scene_id))
            output_h5_path_change = os.path.join(out_cycle_dir_change, "{:0>3}.h5".format(scene_id))

            # 读取原始H5文件中的所有数据
            with h5py.File(output_h5_path, 'r') as f:
                points = f['points'][:]  # 点云数据：3D空间中的采样点坐标
                suction_or = f['suction_or'][:]  # 吸取点法向量：每个点的表面法向量
                suction_seal_scores = f['suction_seal_scores'][:]  # 吸取密封分数：评估密封效果
                suction_wrench_scores = f['suction_wrench_scores'][:]  # 吸取抗扭分数：评估抗扭转能力
                suction_feasibility_scores = f['suction_feasibility_scores'][:]  # 吸取可行性分数：综合可行性评估
                individual_object_size_lable = f['individual_object_size_lable'][:]  # 物体尺寸标签：物体大小分类信息

                # 核心处理：根据法向量与重力方向的夹角调整吸取抗扭分数
                # 这一步骤优化了抗扭分数，使垂直向下的吸取姿态获得更高的评分
                suction_wrench_scores = checke_angles_degrees(suction_or, suction_wrench_scores)

            # 将处理后的数据写入新的H5文件
            with h5py.File(output_h5_path_change, 'w') as f:
                f['points'] = points  # 保存点云数据（未修改）
                f['suction_or'] = suction_or  # 保存法向量数据（未修改）
                f['suction_seal_scores'] = suction_seal_scores  # 保存密封分数（未修改）
                f['suction_wrench_scores'] = suction_wrench_scores  # 保存调整后的抗扭分数（已修改）
                f['suction_feasibility_scores'] = suction_feasibility_scores  # 保存可行性分数（未修改）
                f['individual_object_size_lable'] = individual_object_size_lable  # 保存物体尺寸标签（未修改）
                
            print(f"已完成处理：循环 {cycle_id:04d}，场景 {scene_id:03d}")
    
    print("所有数据处理完成！")



