# -*- coding:utf-8 -*-
"""
本脚本用于计算每个场景中每个物体的单独面积比例, 并将结果保存为csv文件。

@author: Huang Dingtao
@checked: Huang Dingtao

"""

import os
import sys
import re
import argparse

def parse_range_or_single(input_str):
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
parser.add_argument('--cycle_list', type=str, required=True, help='循环编号，支持格式: "5"(单个), "[1,10]"(区间), "{1,3,5}"(列表)')
parser.add_argument('--scene_list', type=str, required=True, help='场景编号，支持格式: "5"(单个), "[1,10]"(区间), "{1,3,5}"(列表)')
FLAGS = parser.parse_args()

# 获取数据集根目录
FILE_DIR = FLAGS.data_dir
# 获取循环次数
CYCLE_LIST = FLAGS.cycle_list
# 获取场景数量
SCENE_LIST = FLAGS.scene_list

import csv
import json
import numpy as np
import shutil
from math import radians
import math
import yaml
import csv
import cv2

# 分割图像的存储路径
OUTDIR_dir_segment_images_single =  os.path.join(FILE_DIR, 'segment_images_single')  # 单物体分割图像
OUTDIR_dir_segment_images =  os.path.join(FILE_DIR, 'segment_images')              # 多物体分割图像

# 单个物体面积比例保存路径
individual_object_size =  os.path.join(FILE_DIR, 'individual_object_size')
if not os.path.exists(individual_object_size):
    os.makedirs(individual_object_size)

def render_scenes():
    """
    遍历所有循环和场景, 计算每个场景中每个物体的单独面积比例, 并保存为csv文件。
    """
    for cycle_id in FLAGS.cycle_list:
        for scene_id in FLAGS.scene_list:
            # 读取当前循环和场景下的多物体分割图像(EXR格式, 包含ID信息)
            img_path = os.path.join(
                OUTDIR_dir_segment_images,
                'cycle_{:0>4}'.format(cycle_id),
                "{:0>3}".format(scene_id),
                'Image0001.exr'
            )
            if not os.path.isfile(img_path):
                print(f"警告：找不到分割图像文件: {img_path}")
                continue  # 跳过本循环

            image_ids = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
            if image_ids is None:
                print(f"警告：无法读取分割图像文件: {img_path}")
                continue  # 跳过本循环

            # 计算所有物体的掩码id(通过分割图像的第二通道归一化得到)
            mask_ids_all = np.round(image_ids[:,:, 1] * (scene_id - 1)).astype('int')

            areas_id = []  # 存储每个物体的面积比例
            for i in range(scene_id):
                # 读取当前物体的单独分割图像
                image_id = cv2.imread(
                    os.path.join(
                        OUTDIR_dir_segment_images_single,
                        'cycle_{:0>4}'.format(cycle_id),
                        "{:0>3}".format(scene_id),
                        "{:0>3}".format(scene_id) + "_{:0>3}".format(i),
                        'Image0001.exr'
                    ),
                    cv2.IMREAD_UNCHANGED
                )
                # 获取当前物体的掩码(第三通道为1的位置为当前物体)
                mask_id = image_id[:,:, 2] == 1
                # 获取所有物体的掩码中属于当前物体的部分
                mask_ids = mask_ids_all == i
                # 只保留当前物体的掩码区域(排除背景或其他物体)
                mask_ids[[image_ids[:,:, 2] != 1]] = 0
                if np.sum(mask_id) != 0:
                    # 计算当前物体的面积比例 = 多物体分割中该物体像素数 / 单物体分割中该物体像素数
                    areas_id.append(np.sum(mask_ids) / np.sum(mask_id))
                else:
                    # 若单物体掩码为0, 则面积比例为0
                    areas_id.append(0)

            # 构建当前循环和场景的保存路径
            save_path = os.path.join(
                individual_object_size,
                'cycle_{:0>4}'.format(cycle_id),
                "{:0>3}".format(scene_id)
            )
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            file_loc = save_path + '/' + '{:0>3}'.format(scene_id) + '.csv'
            assert len(areas_id) == scene_id  # 检查每个场景的物体数量一致
            # 将面积比例写入csv文件, 每一列为一个物体的面积比例
            with open(file_loc, 'w', newline='') as f:
                f_csv = csv.writer(f)
                f_csv.writerow(areas_id)

if __name__ == '__main__':
    import time
    start_time = time.time()
    render_scenes()
    print(time.time() - start_time)
