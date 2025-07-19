import utils
import os
import argparse
import gc
import cv2
import numpy as np
import matplotlib.pyplot as plt
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
from functools import partial
import time

# 命令行参数解析
parser = argparse.ArgumentParser()
# 数据集根目录
parser.add_argument('--data_dir', type=str, default='G:/Diffusion_Suction_DataSet', help='数据集根目录')
parser.add_argument('--cycle_list', type=str, required=True, help='循环编号，支持格式: "5"(单个), "[1,10]"(区间), "{1,3,5}"(列表)')
parser.add_argument('--scene_list', type=str, required=True, help='场景编号，支持格式: "5"(单个), "[1,10]"(区间), "{1,3,5}"(列表)')
parser.add_argument('--max_workers', type=int, default=8, help='线程池最大工作线程数，默认8')
FLAGS = parser.parse_args()

cycle_list = utils.parse_range_or_single(FLAGS.cycle_list)
scene_list = utils.parse_range_or_single(FLAGS.scene_list)
OUTDIR_dir_segment_images_single =  os.path.join(FLAGS.data_dir, 'segment_images_single')  # 单物体分割图像

# 线程锁，用于保护打印输出和文件写入
print_lock = threading.Lock()
error_log_file = os.path.join(os.path.dirname(__file__), 'check_png.txt')

def write_error_to_file(error_message):
    """
    将错误信息写入日志文件
    """
    with print_lock:
        with open(error_log_file, 'a', encoding='utf-8') as f:
            timestamp = time.strftime('%Y-%m-%d %H:%M:%S')
            f.write(f"[{timestamp}] {error_message}\n")

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

def process_single_object(cycle_id, scene_id, object_id):
    """
    处理单个物体的PNG文件检查
    """
    try:
        # 构建文件路径
        filepath = os.path.join(
            OUTDIR_dir_segment_images_single,
            'cycle_{:0>4}'.format(cycle_id),
            "{:0>3}".format(scene_id),
            "{:0>3}".format(scene_id) + "_{:0>3}".format(object_id),
            'Image0001.png'
        )
        
        # 读取PNG mask
        mask_id = read_png_to_numpy(filepath)
        
        # 计算物体在单物体场景下暴露在外的像素和
        exposed_pixels_single = np.sum(mask_id)
        
        if int(exposed_pixels_single) == 0:
            error_msg = f"物体{object_id}在单物体场景{scene_id}中没有暴露在外的像素"
            full_error_msg = f"循环{cycle_id}-场景{scene_id}-物体{object_id}: {error_msg} (文件: {filepath})"
            write_error_to_file(full_error_msg)
            with print_lock:
                print(f"❌ 错误: {error_msg}")
            return (cycle_id, scene_id, object_id, False, error_msg, 0)
        else:
            with print_lock:
                print(f"✅ 循环{cycle_id}-场景{scene_id}-物体{object_id}: {int(exposed_pixels_single)} 像素")
            return (cycle_id, scene_id, object_id, True, "", int(exposed_pixels_single))
            
    except Exception as e:
        error_msg = f"处理文件失败: {str(e)}"
        full_error_msg = f"循环{cycle_id}-场景{scene_id}-物体{object_id}: {error_msg} (文件: {filepath})"
        write_error_to_file(full_error_msg)
        with print_lock:
            print(f"❌ 循环{cycle_id}-场景{scene_id}-物体{object_id}: {error_msg}")
        return (cycle_id, scene_id, object_id, False, error_msg, 0)

def process_scene(cycle_id, scene_id):
    """
    处理单个场景中所有物体的检查
    """
    scene_start_time = time.time()
    tasks = []
    
    # 创建线程池处理当前场景的所有物体
    with ThreadPoolExecutor(max_workers=FLAGS.max_workers) as executor:
        # 提交所有物体的处理任务
        for object_id in range(scene_id):
            future = executor.submit(process_single_object, cycle_id, scene_id, object_id)
            tasks.append(future)
        
        # 收集结果
        success_count = 0
        error_count = 0
        total_pixels = 0
        
        for future in as_completed(tasks):
            cycle_id, scene_id, object_id, success, error_msg, pixels = future.result()
            if success:
                success_count += 1
                total_pixels += pixels
            else:
                error_count += 1
    
    scene_time = time.time() - scene_start_time
    with print_lock:
        print(f"🎯 循环{cycle_id}-场景{scene_id} 完成: {success_count}成功/{error_count}失败, "
              f"总像素: {total_pixels}, 耗时: {scene_time:.2f}秒")
    
    return success_count, error_count, total_pixels

def render_scenes():
    """
    遍历所有循环和场景, 使用线程池并行处理PNG文件检查
    """
    total_start_time = time.time()
    
    # 初始化错误日志文件
    with open(error_log_file, 'w', encoding='utf-8') as f:
        start_time_str = time.strftime('%Y-%m-%d %H:%M:%S')
        f.write(f"PNG文件检查错误日志 - 开始时间: {start_time_str}\n")
        f.write(f"目标目录: {OUTDIR_dir_segment_images_single}\n")
        f.write(f"循环列表: {cycle_list}\n")
        f.write(f"场景列表: {scene_list}\n")
        f.write(f"最大线程数: {FLAGS.max_workers}\n")
        f.write("=" * 80 + "\n\n")
    
    print(f"📊 开始批量检查PNG文件 (最大线程数: {FLAGS.max_workers})")
    print(f"📁 目标目录: {OUTDIR_dir_segment_images_single}")
    print(f"🔄 循环列表: {cycle_list}")
    print(f"🏷️ 场景列表: {scene_list}")
    print(f"📝 错误日志文件: {error_log_file}")
    print("=" * 80)
    
    total_success = 0
    total_error = 0
    total_pixels = 0
    total_scenes = 0
    
    for cycle_id in cycle_list:
        cycle_start_time = time.time()
        
        for scene_id in scene_list:
            success_count, error_count, pixels = process_scene(cycle_id, scene_id)
            total_success += success_count
            total_error += error_count
            total_pixels += pixels
            total_scenes += 1
        
        cycle_time = time.time() - cycle_start_time
        print(f"📈 循环{cycle_id} 完成，耗时: {cycle_time:.2f}秒")
        
        # 执行垃圾回收
        gc.collect()
    
    total_time = time.time() - total_start_time
    
    # 写入最终统计到日志文件
    with open(error_log_file, 'a', encoding='utf-8') as f:
        end_time_str = time.strftime('%Y-%m-%d %H:%M:%S')
        f.write(f"\n" + "=" * 80 + "\n")
        f.write(f"检查完成时间: {end_time_str}\n")
        f.write(f"统计结果:\n")
        f.write(f"  • 总场景数: {total_scenes}\n")
        f.write(f"  • 成功处理: {total_success} 个物体\n")
        f.write(f"  • 失败处理: {total_error} 个物体\n")
        f.write(f"  • 总像素数: {total_pixels:,}\n")
        f.write(f"  • 总耗时: {total_time:.2f}秒\n")
        f.write(f"  • 平均速度: {(total_success + total_error) / total_time:.2f} 个物体/秒\n")
    
    print("=" * 80)
    print(f"🎉 批量检查完成!")
    print(f"📊 统计结果:")
    print(f"   • 总场景数: {total_scenes}")
    print(f"   • 成功处理: {total_success} 个物体")
    print(f"   • 失败处理: {total_error} 个物体")
    print(f"   • 总像素数: {total_pixels:,}")
    print(f"   • 总耗时: {total_time:.2f}秒")
    print(f"   • 平均速度: {(total_success + total_error) / total_time:.2f} 个物体/秒")
    
    if total_error > 0:
        print(f"⚠️  发现 {total_error} 个错误，详细信息已记录到: {error_log_file}")
    else:
        print(f"✅ 所有文件检查通过!")
        # 如果没有错误，在日志文件中也记录一下
        with open(error_log_file, 'a', encoding='utf-8') as f:
            f.write("✅ 所有文件检查通过，无错误发生！\n")
                
if __name__ == '__main__':
    render_scenes()


