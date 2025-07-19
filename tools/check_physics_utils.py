import utils
import os
import argparse
import gc
import csv
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
OUTDIR_dir_physics_result =  os.path.join(FLAGS.data_dir, 'physics_result')  # 物理仿真结果路径

# 线程锁，用于保护打印输出和文件写入
print_lock = threading.Lock()
error_log_file = os.path.join(os.path.dirname(__file__), 'check_physics_utils.txt')

def write_error_to_file(error_message):
    """
    将错误信息写入日志文件
    """
    with print_lock:
        with open(error_log_file, 'a', encoding='utf-8') as f:
            timestamp = time.strftime('%Y-%m-%d %H:%M:%S')
            f.write(f"[{timestamp}] {error_message}\n")
            
def read_csv(csv_path):      
    # 读取csv文件, 返回物体名称、位姿、索引
    with open(csv_path,'r') as csv_file:  
        all_lines=csv.reader(csv_file) 
        list_file = [i for i in all_lines]  # 读取所有行
    array_file = np.array(list_file)[1:]
    obj_name = array_file[:,0]
    obj_index = array_file[:,1].astype('int')
    pose = array_file[:,2:9].astype('float32')
    return obj_name, pose, obj_index


def process_single_object(cycle_id, scene_id, object_id):
    """
    处理单个物体的csv文件检查
    """
    # 设置箱体尺寸：宽800mm, 长600mm, 高500mm, 厚度50mm
    box_width  = 0.8 # 对应x
    box_length = 0.6 # 对应y
    box_thickness = 0.05
    box_height =  0.50
    
    try:
        # 构建文件路径
        csv_path = os.path.join(OUTDIR_dir_physics_result, 'cycle_{:0>4}'.format(cycle_id),"{:0>3}".format(scene_id), "{:0>3}.csv".format(scene_id))
        
        # 读取CSV文件
        obj_name, pose, segment_index = read_csv(csv_path)
        
        # 检查当前物体的位置 (object_id对应的行)
        if object_id >= len(pose):
            error_msg = f"物体{object_id}索引超出范围，CSV中只有{len(pose)}个物体"
            full_error_msg = f"循环{cycle_id}-场景{scene_id}-物体{object_id}: {error_msg} (文件: {csv_path})"
            write_error_to_file(full_error_msg)
            with print_lock:
                print(f"❌ 错误: {error_msg}")
            return (cycle_id, scene_id, object_id, False, error_msg, 0)
            
        translation = pose[object_id, :3]  # 获取当前物体的位置 [x, y, z]
        
        # 检查物体是否超出箱体范围
        x_out_of_range = abs(translation[0]) > (box_width / 2 + 0.3)
        y_out_of_range = abs(translation[1]) > (box_length / 2 + 0.3)
        z_out_of_range = ((abs(translation[2]) > (box_height + 0.3)) or (translation[2] < 0))
        
        if x_out_of_range or y_out_of_range or z_out_of_range:
            error_msg = f"物体{object_id}在场景{scene_id}中超出了箱体范围: 位置({translation[0]:.3f}, {translation[1]:.3f}, {translation[2]:.3f})"
            full_error_msg = f"循环{cycle_id}-场景{scene_id}-物体{object_id}: {error_msg} (文件: {csv_path})"
            write_error_to_file(full_error_msg)
            with print_lock:
                print(f"❌ 错误: {error_msg}")
            return (cycle_id, scene_id, object_id, False, error_msg, 0)
        else:
            with print_lock:
                print(f"✅ 循环{cycle_id}-场景{scene_id}-物体{object_id}: 位置({translation[0]:.3f}, {translation[1]:.3f}, {translation[2]:.3f})")
            return (cycle_id, scene_id, object_id, True, "", 1)
            
    except Exception as e:
        error_msg = f"处理文件失败: {str(e)}"
        full_error_msg = f"循环{cycle_id}-场景{scene_id}-物体{object_id}: {error_msg} (文件: {csv_path})"
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
        total_objects = 0
        
        for future in as_completed(tasks):
            cycle_id, scene_id, object_id, success, error_msg, result = future.result()
            total_objects += 1
            if success:
                success_count += 1
            else:
                error_count += 1
    
    scene_time = time.time() - scene_start_time
    with print_lock:
        print(f"🎯 循环{cycle_id}-场景{scene_id} 完成: {success_count}成功/{error_count}失败, "
              f"总物体数: {total_objects}, 耗时: {scene_time:.2f}秒")
    
    return success_count, error_count

def render_scenes():
    """
    遍历所有循环和场景, 使用线程池并行处理PNG文件检查
    """
    total_start_time = time.time()
    
    # 初始化错误日志文件
    with open(error_log_file, 'w', encoding='utf-8') as f:
        start_time_str = time.strftime('%Y-%m-%d %H:%M:%S')
        f.write(f"物理仿真结果检查错误日志 - 开始时间: {start_time_str}\n")
        f.write(f"目标目录: {OUTDIR_dir_physics_result}\n")
        f.write(f"循环列表: {cycle_list}\n")
        f.write(f"场景列表: {scene_list}\n")
        f.write(f"最大线程数: {FLAGS.max_workers}\n")
        f.write("=" * 80 + "\n\n")
    
    print(f"📊 开始批量检查CSV文件 (最大线程数: {FLAGS.max_workers})")
    print(f"📁 目标目录: {OUTDIR_dir_physics_result}")
    print(f"🔄 循环列表: {cycle_list}")
    print(f"🏷️ 场景列表: {scene_list}")
    print(f"📝 错误日志文件: {error_log_file}")
    print("=" * 80)
    
    total_success = 0
    total_error = 0
    total_scenes = 0
    
    for cycle_id in cycle_list:
        cycle_start_time = time.time()
        
        for scene_id in scene_list:
            success_count, error_count = process_scene(cycle_id, scene_id)
            total_success += success_count
            total_error += error_count
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
        f.write(f"  • 总耗时: {total_time:.2f}秒\n")
        f.write(f"  • 平均速度: {(total_success + total_error) / total_time:.2f} 个物体/秒\n")
    
    print("=" * 80)
    print(f"🎉 批量检查完成!")
    print(f"📊 统计结果:")
    print(f"   • 总场景数: {total_scenes}")
    print(f"   • 成功处理: {total_success} 个物体")
    print(f"   • 失败处理: {total_error} 个物体")
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


