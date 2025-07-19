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
import csv
import json
import numpy as np
import shutil
from math import radians
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
import OpenEXR
import Imath
import cv2
import utils

matplotlib.rcParams['axes.unicode_minus'] = False    # 负号正常显示

font = FontProperties(fname='/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc')

def read_exr_to_numpy(filepath):
    """
    使用OpenEXR读取EXR文件并转换为numpy数组，假定为3通道float32格式。
    """
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

def read_png_to_numpy(filepath):
    """
    使用OpenCV读取PNG文件并转换为mask数组。
    """
    # 读取PNG图像
    png_file = cv2.imread(filepath, cv2.IMREAD_UNCHANGED)
    
    if png_file is None:
        raise ValueError(f"无法读取PNG文件: {filepath}")
    
    # 如果是单通道图像(灰度图)，直接作为mask
    if len(png_file.shape) == 2:
        mask = png_file
    # 如果是多通道图像，转换为灰度图
    elif len(png_file.shape) == 3:
        if png_file.shape[2] == 3:  # RGB图像
            mask = cv2.cvtColor(png_file, cv2.COLOR_BGR2GRAY)
        elif png_file.shape[2] == 4:  # RGBA图像
            mask = cv2.cvtColor(png_file, cv2.COLOR_BGRA2GRAY)
        else:
            # 其他情况，取第一个通道
            mask = png_file[:, :, 0]
    else:
        raise ValueError(f"不支持的图像格式: {png_file.shape}")
    
    # 将mask转换为float32格式，并归一化到0-1范围
    mask = mask.astype(np.float32)
    if mask.max() > 1.0:
        mask = mask / 255.0
    
    # 生成二值mask：大于0.5的像素为True，其余为False
    mask = mask > 0.5
    
    return mask

# 命令行参数解析
parser = argparse.ArgumentParser()
# 数据集根目录
parser.add_argument('--data_dir', type=str, default='G:/Diffusion_Suction_DataSet', help='数据集根目录')
parser.add_argument('--cycle_list', type=str, required=True, help='循环编号，支持格式: "5"(单个), "[1,10]"(区间), "{1,3,5}"(列表)')
parser.add_argument('--scene_list', type=str, required=True, help='场景编号，支持格式: "5"(单个), "[1,10]"(区间), "{1,3,5}"(列表)')
FLAGS = parser.parse_args()

# 获取数据集根目录
FILE_DIR = FLAGS.data_dir
# 获取循环和场景列表
cycle_list = utils.parse_range_or_single(FLAGS.cycle_list)
scene_list = utils.parse_range_or_single(FLAGS.scene_list)


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
    for cycle_id in cycle_list:
        for scene_id in scene_list:
            # 读取当前循环和场景下的多物体分割图像(EXR格式, 包含ID信息)
            image_ids = read_exr_to_numpy(
                os.path.join(
                    OUTDIR_dir_segment_images,
                    'cycle_{:0>4}'.format(cycle_id),
                    "{:0>3}".format(scene_id),
                    'Image0001.exr'
                )
            )
            # 计算所有物体的掩码id(通过分割图像的第二通道归一化得到)
            # mask_ids_all = np.round(image_ids[:,:, 1] * (scene_id - 1)).astype('int')
            # 先将image_ids[:,:, 1]中偏小的值全部置为0.0
            step = 1 / scene_id
            mask_ids_all = np.full(image_ids[:,:,1].shape, 255, dtype=np.float32)
            valid_mask = image_ids[:,:,0] >= 0.5
            quotient, remainder = np.divmod(image_ids[:,:,1], step)
            mask_ids_all[valid_mask] = quotient[valid_mask]

            # 可视化
            # plt.figure()
            # plt.imshow(mask_ids_all, cmap='tab20')
            # plt.title(f"mask_ids_all for scene {scene_id}")
            # plt.colorbar()
            # plt.show()

            areas_id = []  # 存储每个物体的面积比例
            for i in range(scene_id):
                # 读取当前物体的单独分割图像
                mask_id = read_png_to_numpy(
                    os.path.join(
                        OUTDIR_dir_segment_images_single,
                        'cycle_{:0>4}'.format(cycle_id),
                        "{:0>3}".format(scene_id),
                        "{:0>3}".format(scene_id) + "_{:0>3}".format(i),
                        'Image0001.png'
                    )
                )
                # 获取当前物体的掩码(第三通道为1的位置为当前物体)
                # mask_id = image_id[:,:, 2] = 1
                # mask_id = (image_id[:,:, 1] >= 0.0) & (image_id[:,:, 0] <= 1 / scene_id + 0.00000001)
                # mask_id = (image_id[:,:, 1] < 0.05) & (image_id[:,:, 0] > 0.5)

                # 获取所有物体的掩码中属于当前物体的部分
                mask_ids = mask_ids_all == i

                # 计算物体在多物体场景下暴露在外的像素和
                exposed_pixels = np.sum(mask_ids)

                # 计算物体在单物体场景下暴露在外的像素和
                exposed_pixels_single = np.sum(mask_id)

                # 计算交集
                intersection = np.sum(mask_id & mask_ids)

                # fig, axs = plt.subplots(1, 3, figsize=(12, 4))
                # axs[0].imshow(mask_id, cmap='gray')
                # axs[0].set_title(f"mask_id (单物体掩码)", fontproperties=font)
                # axs[1].imshow(mask_ids, cmap='gray')
                # axs[1].set_title(f"mask_ids (多物体分割ID=={i})", fontproperties=font)
                # # 叠加显示，两者都为True的像素显示为红色
                # overlay = np.zeros((*mask_id.shape, 3), dtype=np.float32)
                # overlay[mask_id & mask_ids] = [1, 0, 0]  # 红色
                # overlay[mask_id & ~mask_ids] = [0, 1, 0] # 绿色
                # overlay[~mask_id & mask_ids] = [0, 0, 1] # 蓝色
                # axs[2].imshow(overlay)
                # axs[2].set_title("重叠区域: 红=都为True, 绿=仅mask_id, 蓝=仅mask_ids", fontproperties=font)
                # for ax in axs:
                #     ax.axis('off')
                # plt.suptitle(f"scene {scene_id}, object {i}", fontproperties=font)
                # plt.tight_layout()
                # plt.show()

                if int(exposed_pixels_single) * 1.5 < int(exposed_pixels):
                    fig, axs = plt.subplots(1, 3, figsize=(12, 4))
                    axs[0].imshow(mask_id, cmap='gray')
                    axs[0].set_title(f"mask_id (单物体掩码)", fontproperties=font)
                    axs[1].imshow(mask_ids, cmap='gray')
                    axs[1].set_title(f"mask_ids (多物体分割ID=={i})", fontproperties=font)
                    # 叠加显示，两者都为True的像素显示为红色
                    overlay = np.zeros((*mask_id.shape, 3), dtype=np.float32)
                    overlay[mask_id & mask_ids] = [1, 0, 0]  # 红色
                    overlay[mask_id & ~mask_ids] = [0, 1, 0] # 绿色
                    overlay[~mask_id & mask_ids] = [0, 0, 1] # 蓝色
                    axs[2].imshow(overlay)
                    axs[2].set_title("重叠区域: 红=都为True, 绿=仅mask_id, 蓝=仅mask_ids", fontproperties=font)
                    for ax in axs:
                        ax.axis('off')
                    plt.suptitle(f"scene {scene_id}, object {i}", fontproperties=font)
                    plt.tight_layout()
                    plt.show()

                if exposed_pixels_single == 0:
                    print(f"循环 {cycle_id} ，场景 {scene_id} 中：物体 {i} 的单物体掩码像素数为0, 跳过面积比例计算")
                    areas_id.append(0)
                    raise ValueError(f"循环 {cycle_id} ，场景 {scene_id} 中：物体 {i} 的单物体掩码像素数为0, 无法计算面积比例")
                else:
                    proportion = intersection / exposed_pixels_single
                    areas_id.append(proportion)
                    print(f"循环 {cycle_id} ，场景 {scene_id} 中：物体 {i} 面积比例: {proportion:.4f}")

                # # 只保留当前物体的掩码区域(排除背景或其他物体)
                # mask_ids[image_ids[:,:, 2] != 1] = 0

                # mask_only = np.sum(mask_id)
                # mask_in_scence =  np.sum(mask_ids)
                # if mask_only != 0:
                #     # 计算当前物体的面积比例 = 多物体分割中该物体像素数 / 单物体分割中该物体像素数
                #     print(f"循环 {cycle_id} ，场景 {scene_id} 中：物体 {i} 的多物体掩码像素数: {mask_in_scence}")
                #     print(f"循环 {cycle_id} ，场景 {scene_id} 中：物体 {i} 的单物体掩码像素数: {mask_only}")
                #     if mask_in_scence == 0:
                #         print(f"循环 {cycle_id} ，场景 {scene_id} 中：物体 {i} 的多物体掩码像素数为0, 跳过面积比例计算")
                #         raise ValueError(f"循环 {cycle_id} ，场景 {scene_id} 中：物体 {i} 的多物体掩码像素数为0, 无法计算面积比例")
                #     proportion = np.sum(mask_ids) / np.sum(mask_id)
                #     areas_id.append(proportion)
                #     print(f"循环 {cycle_id} ，场景 {scene_id} 中：物体 {i} 面积比例: {proportion:.4f}")
                # else:
                #     # 若单物体掩码为0, 则面积比例为0
                #     areas_id.append(0)
                #     print(f"循环 {cycle_id} ，场景 {scene_id} 中：物体 {i} 面积比例: 0.0000 (单物体掩码为0)")

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
    
    # # 测试PNG读取函数
    # test_file = "/home/lixinlong/Data/Diffusion_Suction_DataSet/segment_images_single/cycle_0002/049/049_000/Image0001.png"
    
    # print(f"正在读取文件: {test_file}")
    
    # try:
    #     # 读取PNG mask
    #     mask = read_png_to_numpy(test_file)
        
    #     print(f"读取成功!")
    #     print(f"Mask形状: {mask.shape}")
    #     print(f"Mask数据类型: {mask.dtype}")
    #     print(f"Mask取值范围: {mask.min()} ~ {mask.max()}")
    #     print(f"True像素数: {np.sum(mask)}")
    #     print(f"False像素数: {np.sum(~mask)}")
    #     print(f"总像素数: {mask.size}")
        
    #     # 可视化mask
    #     plt.figure(figsize=(12, 5))
        
    #     # 原始读取的图像（用于对比）
    #     plt.subplot(1, 2, 1)
    #     original_img = cv2.imread(test_file, cv2.IMREAD_UNCHANGED)
    #     if original_img is not None:
    #         if len(original_img.shape) == 3:
    #             original_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
    #         plt.imshow(original_img, cmap='gray')
    #         plt.title('原始PNG图像', fontproperties=font)
    #     else:
    #         plt.text(0.5, 0.5, '无法显示原始图像', ha='center', va='center')
    #         plt.title('原始PNG图像 (读取失败)', fontproperties=font)
    #     plt.axis('off')
        
    #     # 处理后的二值mask
    #     plt.subplot(1, 2, 2)
    #     plt.imshow(mask, cmap='gray')
    #     plt.title(f'二值Mask\n(True: {np.sum(mask)}, False: {np.sum(~mask)})', fontproperties=font)
    #     plt.axis('off')
        
    #     plt.tight_layout()
    #     plt.show()
        
    #     # 显示mask的统计信息
    #     print("\n=== Mask统计信息 ===")
    #     print(f"图像尺寸: {mask.shape[1]} x {mask.shape[0]}")
    #     print(f"前景像素比例: {np.sum(mask) / mask.size * 100:.2f}%")
    #     print(f"背景像素比例: {np.sum(~mask) / mask.size * 100:.2f}%")
        
    # except Exception as e:
    #     print(f"读取PNG文件时发生错误: {e}")
    #     import traceback
    #     traceback.print_exc()
