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
import gc
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
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
from datetime import datetime

matplotlib.rcParams['axes.unicode_minus'] = False    # 负号正常显示

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
parser.add_argument('--num_workers', type=int, default=8, help='线程池的工作线程数 (默认: 8)')
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

# 线程安全的打印锁
print_lock = threading.Lock()

def thread_safe_print(message):
    """线程安全的打印函数"""
    with print_lock:
        timestamp = datetime.now().strftime("%H:%M:%S")
        print(f"[{timestamp}] {message}")

def process_single_cycle_scene(cycle_id, scene_id):
    """
    处理单个循环-场景组合，计算每个物体的面积比例
    
    Args:
        cycle_id (int): 循环编号
        scene_id (int): 场景编号
        
    Returns:
        tuple: (cycle_id, scene_id, success, error_msg)
    """
    try:
        thread_safe_print(f"开始处理循环 {cycle_id}，场景 {scene_id}")
        
        # 读取当前循环和场景下的多物体分割图像(EXR格式, 包含ID信息)
        image_ids = read_exr_to_numpy(
            os.path.join(
                OUTDIR_dir_segment_images,
                'cycle_{:0>4}'.format(cycle_id),
                "{:0>3}".format(scene_id),
                'Image0001.exr'
            )
        )
        
        # 计算所有物体的掩码id
        step = 1 / scene_id
        mask_ids_all = np.full(image_ids[:,:,1].shape, 255, dtype=np.float32)
        valid_mask = image_ids[:,:,0] >= 0.5
        quotient, remainder = np.divmod(image_ids[:,:,1], step)
        mask_ids_all[valid_mask] = quotient[valid_mask]

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

            # 获取所有物体的掩码中属于当前物体的部分
            mask_ids = mask_ids_all == i

            # 计算物体在多物体场景下暴露在外的像素和
            exposed_pixels = np.sum(mask_ids)

            # 计算物体在单物体场景下暴露在外的像素和
            exposed_pixels_single = np.sum(mask_id)

            # 计算交集
            intersection = np.sum(mask_id & mask_ids)

            if exposed_pixels_single == 0:
                thread_safe_print(f"循环 {cycle_id}，场景 {scene_id} 中：物体 {i} 的单物体掩码像素数为0, 跳过面积比例计算")
                areas_id.append(0)
                raise ValueError(f"循环 {cycle_id}，场景 {scene_id} 中：物体 {i} 的单物体掩码像素数为0, 无法计算面积比例")
            else:
                proportion = intersection / exposed_pixels_single
                areas_id.append(proportion)
                # 只在比例异常时打印详细信息
                if proportion < 0.1 or proportion > 1.1:
                    thread_safe_print(f"⚠️ 循环 {cycle_id}，场景 {scene_id} 中：物体 {i} 面积比例异常: {proportion:.4f}")

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
        
        # 将面积比例写入csv文件
        with open(file_loc, 'w', newline='') as f:
            f_csv = csv.writer(f)
            f_csv.writerow(areas_id)
            
        thread_safe_print(f"✅ 完成处理循环 {cycle_id}，场景 {scene_id}")
        gc.collect()  # 释放内存
        return (cycle_id, scene_id, True, None)
        
    except Exception as e:
        error_msg = f"循环 {cycle_id}，场景 {scene_id} 处理失败: {str(e)}"
        thread_safe_print(f"❌ {error_msg}")
        return (cycle_id, scene_id, False, error_msg)



def render_scenes():
    """
    使用线程池遍历所有循环和场景，计算每个场景中每个物体的单独面积比例，并保存为csv文件。
    """
    # 生成所有循环-场景组合
    tasks = []
    for cycle_id in cycle_list:
        for scene_id in scene_list:
            tasks.append((cycle_id, scene_id))
    
    total_tasks = len(tasks)
    completed_tasks = 0
    failed_tasks = []
    
    thread_safe_print(f"🚀 开始处理 {total_tasks} 个循环-场景组合，使用 {FLAGS.num_workers} 个工作线程")
    
    # 使用线程池并行处理
    with ThreadPoolExecutor(max_workers=FLAGS.num_workers) as executor:
        # 提交所有任务
        future_to_task = {
            executor.submit(process_single_cycle_scene, cycle_id, scene_id): (cycle_id, scene_id)
            for cycle_id, scene_id in tasks
        }
        
        # 处理完成的任务
        for future in as_completed(future_to_task):
            cycle_id, scene_id = future_to_task[future]
            try:
                result_cycle_id, result_scene_id, success, error_msg = future.result()
                completed_tasks += 1
                
                if success:
                    # 计算进度
                    progress = completed_tasks / total_tasks * 100
                    thread_safe_print(f"📊 进度: {completed_tasks}/{total_tasks} ({progress:.1f}%) - 循环{result_cycle_id}-场景{result_scene_id}")
                else:
                    failed_tasks.append((result_cycle_id, result_scene_id, error_msg))
                    
            except Exception as e:
                completed_tasks += 1
                error_msg = f"任务执行异常: {str(e)}"
                failed_tasks.append((cycle_id, scene_id, error_msg))
                thread_safe_print(f"❌ 循环 {cycle_id}，场景 {scene_id} 执行异常: {e}")
    
    # 输出最终统计
    thread_safe_print(f"\n📈 处理完成统计:")
    thread_safe_print(f"   总任务数: {total_tasks}")
    thread_safe_print(f"   成功任务: {completed_tasks - len(failed_tasks)}")
    thread_safe_print(f"   失败任务: {len(failed_tasks)}")
    
    if failed_tasks:
        thread_safe_print(f"\n❌ 失败的任务列表:")
        for cycle_id, scene_id, error_msg in failed_tasks:
            thread_safe_print(f"   循环{cycle_id}-场景{scene_id}: {error_msg}")
        
        # 将失败的任务保存到文件
        error_log_file = os.path.join(individual_object_size, 'processing_errors.txt')
        with open(error_log_file, 'w', encoding='utf-8') as f:
            f.write(f"处理时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"失败任务数: {len(failed_tasks)}\n\n")
            for cycle_id, scene_id, error_msg in failed_tasks:
                f.write(f"循环{cycle_id}-场景{scene_id}: {error_msg}\n")
        thread_safe_print(f"📝 失败任务已记录到: {error_log_file}")
    else:
        thread_safe_print(f"🎉 所有任务均成功完成!")

if __name__ == '__main__':
    import time
    start_time = time.time()
    
    print(f"🔧 配置信息:")
    print(f"   数据目录: {FILE_DIR}")
    print(f"   循环列表: {cycle_list}")
    print(f"   场景列表: {scene_list}")
    print(f"   工作线程数: {FLAGS.num_workers}")
    print(f"   总任务数: {len(cycle_list) * len(scene_list)}")
    print()
    
    render_scenes()
    
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"\n⏱️ 总耗时: {elapsed_time:.2f} 秒 ({elapsed_time/60:.1f} 分钟)")
    
    # 计算平均每个任务的处理时间
    total_tasks = len(cycle_list) * len(scene_list)
    avg_time_per_task = elapsed_time / total_tasks
    print(f"📊 平均每个循环-场景组合处理时间: {avg_time_per_task:.2f} 秒")
    
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
