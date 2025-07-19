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
import gc
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
from datetime import datetime
import cv2
    
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
parser.add_argument('--num_workers', type=int, default=4, help='线程池的工作线程数 (默认: 4，建议不超过GPU内存限制)')
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

# 线程安全的打印锁
print_lock = threading.Lock()

def thread_safe_print(message):
    """线程安全的打印函数"""
    with print_lock:
        timestamp = datetime.now().strftime("%H:%M:%S")
        print(f"[{timestamp}] {message}")

def process_single_cycle_scene(cycle_id, scene_id, data_generator_params):
    """
    处理单个循环-场景组合，生成H5训练数据
    
    Args:
        cycle_id (int): 循环编号
        scene_id (int): 场景编号
        data_generator_params (dict): H5数据生成器的参数配置
        
    Returns:
        tuple: (cycle_id, scene_id, success, error_msg)
    """
    try:
        thread_safe_print(f"开始处理循环 {cycle_id}，场景 {scene_id}")
        
        # 为每个线程创建独立的H5数据生成器实例，避免线程冲突
        g = H5DataGenerator(
            params_file_name=data_generator_params['parameter_file'], 
            camera_info_file_name=data_generator_params['camera_info_file'], 
            objs_path=data_generator_params['objs_path'],
            target_num_point=16384,
            test_flag=False
        )
        
        # 1. 构建深度图像文件路径并加载
        depth_image_path = os.path.join(
            data_generator_params['depth_dir'], 
            'cycle_{:0>4}'.format(cycle_id), 
            "{:0>3}".format(scene_id), 
            'Image0001.png'
        )
        depth_image = cv2.imread(depth_image_path, cv2.IMREAD_UNCHANGED)
        if depth_image is None:
            raise ValueError(f"无法读取深度图像文件: {depth_image_path}")
    
        # 2. 构建分割图像文件路径并加载
        seg_img_path = os.path.join(
            data_generator_params['segment_dir'], 
            'cycle_{:0>4}'.format(cycle_id), 
            "{:0>3}".format(scene_id), 
            'Image0001.exr'
        )
        segment_image = read_exr_to_numpy(seg_img_path)

        # 3. 构建真值标签文件路径
        gt_file_path = os.path.join(
            data_generator_params['gt_path'], 
            'cycle_{:0>4}'.format(cycle_id), 
            "{:0>3}".format(scene_id), 
            '{:0>3}.csv'.format(scene_id)
        )

        # 4. 构建输出H5文件路径
        out_cycle_dir = os.path.join(
            data_generator_params['train_set_dir'], 
            'cycle_{:0>4}'.format(cycle_id)
        )
        if not os.path.exists(out_cycle_dir):
            os.makedirs(out_cycle_dir, exist_ok=True)
        
        output_h5_path = os.path.join(out_cycle_dir, "{:0>3}.h5".format(scene_id))

        # 5. 构建单个物体尺寸标签文件路径
        individual_object_size_path = os.path.join(
            data_generator_params['individual_path'], 
            'cycle_{:0>4}'.format(cycle_id), 
            "{:0>3}".format(scene_id), 
            '{:0>3}.csv'.format(scene_id)
        )

        # 6. 核心处理步骤：调用数据生成器处理当前场景
        g.process_train_set(
            depth_image, segment_image, gt_file_path, 
            output_h5_path, individual_object_size_path
        )
        
        thread_safe_print(f"✅ 完成处理循环 {cycle_id}，场景 {scene_id}")
        gc.collect()  # 手动触发垃圾回收，防止内存泄漏
        return (cycle_id, scene_id, True, None)
        
    except Exception as e:
        error_msg = f"循环 {cycle_id}，场景 {scene_id} 处理失败: {str(e)}"
        thread_safe_print(f"❌ {error_msg}")
        return (cycle_id, scene_id, False, error_msg)

if __name__ == "__main__":
    import time
    start_time = time.time()
    
    # 解析循环和场景编号为整数列表
    CYCLE_idx_list = parse_range_or_single(FLAGS.cycle_list)
    SCENE_idx_list = parse_range_or_single(FLAGS.scene_list)

    # 准备数据生成器参数配置
    data_generator_params = {
        'parameter_file': FLAGS.parameter_file,
        'camera_info_file': FLAGS.camera_info_file,
        'objs_path': os.path.join(FLAGS.data_dir, 'OBJ'),
        'depth_dir': DEPTH_DIR,
        'segment_dir': SEGMENT_DIR,
        'gt_path': GT_PATH,
        'train_set_dir': TRAIN_SET_DIR,
        'individual_path': INDIVIDUA_PATH
    }
    
    # 生成所有循环-场景组合
    tasks = []
    for cycle_id in CYCLE_idx_list:
        for scene_id in SCENE_idx_list:
            tasks.append((cycle_id, scene_id))
    
    total_tasks = len(tasks)
    completed_tasks = 0
    failed_tasks = []
    
    print(f"🔧 配置信息:")
    print(f"   数据目录: {FLAGS.data_dir}")
    print(f"   循环列表: {CYCLE_idx_list}")
    print(f"   场景列表: {SCENE_idx_list}")
    print(f"   工作线程数: {FLAGS.num_workers}")
    print(f"   总任务数: {total_tasks}")
    print()
    
    thread_safe_print(f"🚀 开始处理 {total_tasks} 个循环-场景组合，使用 {FLAGS.num_workers} 个工作线程")
    
    # 使用线程池并行处理
    with ThreadPoolExecutor(max_workers=FLAGS.num_workers) as executor:
        # 提交所有任务
        future_to_task = {
            executor.submit(process_single_cycle_scene, cycle_id, scene_id, data_generator_params): (cycle_id, scene_id)
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
                    thread_safe_print(f"📊 进度: {completed_tasks}/{total_tasks} ({progress:.1f}%) - 循环{result_cycle_id:04d}-场景{result_scene_id:03d}")
                else:
                    failed_tasks.append((result_cycle_id, result_scene_id, error_msg))
                    
            except Exception as e:
                completed_tasks += 1
                error_msg = f"任务执行异常: {str(e)}"
                failed_tasks.append((cycle_id, scene_id, error_msg))
                thread_safe_print(f"❌ 循环 {cycle_id}，场景 {scene_id} 执行异常: {e}")
    
    # 输出最终统计
    end_time = time.time()
    elapsed_time = end_time - start_time
    
    thread_safe_print(f"\n📈 处理完成统计:")
    thread_safe_print(f"   总任务数: {total_tasks}")
    thread_safe_print(f"   成功任务: {completed_tasks - len(failed_tasks)}")
    thread_safe_print(f"   失败任务: {len(failed_tasks)}")
    thread_safe_print(f"   总耗时: {elapsed_time:.2f} 秒 ({elapsed_time/60:.1f} 分钟)")
    
    if total_tasks > 0:
        avg_time_per_task = elapsed_time / total_tasks
        thread_safe_print(f"   平均每任务: {avg_time_per_task:.2f} 秒")
    
    if failed_tasks:
        thread_safe_print(f"\n❌ 失败的任务列表:")
        for cycle_id, scene_id, error_msg in failed_tasks:
            thread_safe_print(f"   循环{cycle_id:04d}-场景{scene_id:03d}: {error_msg}")
        
        # 将失败的任务保存到文件
        error_log_file = os.path.join(OUT_ROOT_DIR, 'generation_errors.txt')
        with open(error_log_file, 'w', encoding='utf-8') as f:
            f.write(f"处理时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"失败任务数: {len(failed_tasks)}\n\n")
            for cycle_id, scene_id, error_msg in failed_tasks:
                f.write(f"循环{cycle_id:04d}-场景{scene_id:03d}: {error_msg}\n")
        thread_safe_print(f"📝 失败任务已记录到: {error_log_file}")
    else:
        thread_safe_print(f"🎉 所有训练数据集生成完成！")

