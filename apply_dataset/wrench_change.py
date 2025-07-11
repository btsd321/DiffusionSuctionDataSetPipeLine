"""

@author: Huang Dingtao
@checked: Huang Dingtao


"""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
# import torch
# test = torch.randn(30, 3).cuda()

import sys
FILE_PATH = os.path.abspath(__file__)
FILE_DIR_generate_dataset = os.path.dirname(FILE_PATH)
FILE_DIR = os.path.dirname(FILE_DIR_generate_dataset)

sys.path.append(FILE_DIR)
DATASET_apply_dataset_parameter = os.path.join(FILE_DIR_generate_dataset, 'parameter.json')
    

from H5DataGenerator import *
OUT_ROOT_DIR =  os.path.join(FILE_DIR, 'h5_dataset')
TRAIN_SET_DIR = os.path.join(OUT_ROOT_DIR, 'train')

OUT_ROOT_DIR_Change =  os.path.join(FILE_DIR, 'h5_dataset_change')
if not os.path.exists(OUT_ROOT_DIR_Change):
    os.makedirs(OUT_ROOT_DIR_Change)    
TRAIN_SET_DIR_Change = os.path.join(OUT_ROOT_DIR_Change, 'train')
if not os.path.exists( TRAIN_SET_DIR_Change ):
    os.mkdir(TRAIN_SET_DIR_Change)

def checke_angles_degrees(normals,score):
    """
    根据法向量与参考向量(0,0,-1)之间的夹角, 对分数进行加权调整。

    参数:
        normals: 法向量数组, 形状为(N, 3)
        score: 原始分数数组

    返回:
        score: 加权调整后的分数数组
    """

    reference_vector = np.array([0, 0, -1])  # 参考向量, 指向负Z轴

    dot_products = np.sum(normals * reference_vector, axis=1)  # 计算法向量与参考向量的点积

    norm_magnitudes = np.linalg.norm(normals, axis=1)  # 计算每个法向量的模长
    reference_magnitude = np.linalg.norm(reference_vector)  # 计算参考向量的模长

    cos_angles = dot_products / (norm_magnitudes * reference_magnitude)  # 计算夹角的余弦值

    cos_angles = np.clip(cos_angles, -1.0, 1.0)  # 防止数值溢出, 限制在[-1, 1]范围内

    angles = np.arccos(cos_angles)  # 计算夹角(弧度)

    angles_degrees = np.degrees(angles)  # 转换为角度

    mapped_values = 1 - (angles_degrees / 180)  # 将角度映射到[0,1]区间, 角度越小值越大

    score *= mapped_values  # 用映射值对原分数加权
    return score

if __name__ == "__main__":
    CYCLE_idx_list = range(0, 1)
    SCENE_idx_list = range(1, 51)

    g = H5DataGenerator(DATASET_apply_dataset_parameter)
    for cycle_id in CYCLE_idx_list:
        out_cycle_dir = os.path.join(TRAIN_SET_DIR, 'cycle_{:0>4}'.format(cycle_id))
        out_cycle_dir_change = os.path.join(TRAIN_SET_DIR_Change, 'cycle_{:0>4}'.format(cycle_id))
        
        if not os.path.exists(out_cycle_dir_change):
            os.mkdir(out_cycle_dir_change)
        for scene_id in SCENE_idx_list:
            # 定义输入和输出h5文件路径
            output_h5_path = os.path.join(out_cycle_dir, "{:0>3}.h5".format(scene_id))
            output_h5_path_change  = os.path.join(out_cycle_dir_change , "{:0>3}.h5".format(scene_id))

            with h5py.File(output_h5_path, 'r') as f:
                points = f['points'][:]  # 点云数据
                suction_or = f['suction_or'][:]  # 吸取点法向量
                suction_seal_scores = f['suction_seal_scores'][:]  # 吸取密封分数
                suction_wrench_scores = f['suction_wrench_scores'][:]  # 吸取抗扭分数
                suction_feasibility_scores = f['suction_feasibility_scores'][:]  # 吸取可行性分数
                individual_object_size_lable = f['individual_object_size_lable'][:]  # 物体尺寸标签

                # 根据法向量与参考向量夹角调整吸取抗扭分数
                suction_wrench_scores = checke_angles_degrees(suction_or,suction_wrench_scores)

            with h5py.File(output_h5_path_change,'w') as f:
                f['points'] = points
                f['suction_or'] = suction_or
                f['suction_seal_scores'] = suction_seal_scores
                f['suction_wrench_scores'] = suction_wrench_scores
                f['suction_feasibility_scores'] = suction_feasibility_scores
                f['individual_object_size_lable'] = individual_object_size_lable



