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
if not os.path.exists(OUT_ROOT_DIR):
    os.makedirs(OUT_ROOT_DIR)    

TRAIN_SET_DIR = os.path.join(OUT_ROOT_DIR, 'train')
if not os.path.exists( TRAIN_SET_DIR ):
    os.mkdir(TRAIN_SET_DIR)

# 定义各类数据的目录
GT_DIR = os.path.join(FILE_DIR, 'gt')  # 真值(物体位姿等)目录
SEGMENT_DIR = os.path.join(FILE_DIR, 'segment_images')  # 分割图像目录
DEPTH_DIR = os.path.join(FILE_DIR, 'depth_images')  # 深度图像目录
OBJ_PATH = os.path.join(FILE_DIR, 'OBJ')  # 物体模型目录
GT_PATH =  os.path.join(FILE_DIR, 'gt')  # 真值目录
INDIVIDUA_PATH =  os.path.join(FILE_DIR, 'individual_object_size')  # 单个物体尺寸标签目录

if __name__ == "__main__":
    # 设置循环和场景编号范围
    CYCLE_idx_list = range(81, 100)
    SCENE_idx_list = range(1, 51)

    # --------------------------------------------------------------------------check estimate_normals
    # 用于调试法线估计时只处理一个循环
    CYCLE_idx_list = range(0, 1)
    SCENE_idx_list = range(1, 51)
    # --------------------------------------------------------------------------check estimate_normals

    # 实例化数据生成器
    g = H5DataGenerator(DATASET_apply_dataset_parameter)
    for cycle_id in CYCLE_idx_list:
        out_cycle_dir = os.path.join(TRAIN_SET_DIR, 'cycle_{:0>4}'.format(cycle_id))
        if not os.path.exists(out_cycle_dir):
            os.mkdir(out_cycle_dir)
      
        for scene_id in SCENE_idx_list:
            # 加载深度图像
            depth_image_path = os.path.join(DEPTH_DIR,'cycle_{:0>4}'.format(cycle_id), "{:0>3}".format(scene_id),'Image0001.png')
            depth_image = cv2.imread(depth_image_path,cv2.IMREAD_UNCHANGED)
        
            # 加载分割图像
            seg_img_path = os.path.join(SEGMENT_DIR,'cycle_{:0>4}'.format(cycle_id), "{:0>3}".format(scene_id),'Image0001.exr')
            segment_image = cv2.imread(seg_img_path,cv2.IMREAD_UNCHANGED)

            # 定义真值(物体位姿等)csv文件路径
            gt_file_path = os.path.join(GT_PATH,'cycle_{:0>4}'.format(cycle_id),"{:0>3}".format(scene_id))+'/' + '{:0>3}'.format(scene_id) + '.csv'

            # 定义输出h5文件路径
            output_h5_path = os.path.join(out_cycle_dir, "{:0>3}.h5".format(scene_id)    )

            # 加载稠密点云(可选, 注释掉)
            # dense_point_path =  os.path.join(OBJ_PATH + '_{}'.format(obj_id) '_{}'.format(obj_id)+".ply")

            # 加载单个物体的抓取分数用来knn得到seal分数(可选, 注释掉)
            # seal_path = os.path.join(FILE_DIR, 'OBJ' + '_{}'.format(obj_id),   '_{}'.format(obj_id) + ".npz")

            # 加载单个物体的尺寸标签csv路径
            individual_object_size_path =  os.path.join(INDIVIDUA_PATH, 'cycle_{:0>4}'.format(cycle_id),"{:0>3}".format(scene_id))+'/' + '{:0>3}'.format(scene_id) + '.csv'

            # 处理当前场景, 生成h5数据集
            g.process_train_set( depth_image, segment_image, gt_file_path, output_h5_path,individual_object_size_path)

