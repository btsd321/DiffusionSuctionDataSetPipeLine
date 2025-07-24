# -*- coding:utf-8 -*-
"""
本文件用于在Blender中批量渲染三维场景, 自动导入物体模型、设置相机与光源参数, 并输出RGB图像、深度图和分割标签图。适用于数据集的自动生成与仿真渲染流程。
@author: Huang Dingtao
@checked: Huang Dingtao
"""

import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'config'))
from camera_info import CameraInfo
import OpenEXR
import Imath
import argparse
import re
import gc
import multiprocessing as mp
import subprocess
import time
import threading

def monitor_gpu_usage(gpu_id, duration=30, interval=2):
    """监控指定GPU的使用情况"""
    print(f"开始监控GPU {gpu_id}...")
    
    start_time = time.time()
    max_util = 0
    max_mem = 0
    
    while time.time() - start_time < duration:
        try:
            # 获取详细的GPU信息
            result = subprocess.run([
                'nvidia-smi', '-i', str(gpu_id), 
                '--query-gpu=utilization.gpu,utilization.memory,memory.used,memory.total,temperature.gpu,power.draw',
                '--format=csv,noheader,nounits'
            ], capture_output=True, text=True, check=True)
            
            gpu_util, mem_util, mem_used, mem_total, temp, power = result.stdout.strip().split(', ')
            
            # 转换数值
            gpu_util = float(gpu_util) if gpu_util != 'N/A' else 0
            mem_util = float(mem_util) if mem_util != 'N/A' else 0
            mem_used = int(mem_used) if mem_used != 'N/A' else 0
            temp = float(temp) if temp != 'N/A' else 0
            power = float(power) if power != 'N/A' else 0
            
            max_util = max(max_util, gpu_util)
            max_mem = max(max_mem, mem_used)
            
            print(f"[GPU {gpu_id}] 利用率: {gpu_util:5.1f}% | 显存利用率: {mem_util:5.1f}% | "
                  f"显存使用: {mem_used}MB | 温度: {temp}°C | 功耗: {power}W")
            
            # 检查计算进程
            proc_result = subprocess.run([
                'nvidia-smi', '-i', str(gpu_id), '--query-compute-apps=pid,process_name,used_memory',
                '--format=csv,noheader'
            ], capture_output=True, text=True)
            
            if proc_result.stdout.strip():
                print(f"[GPU {gpu_id}] 计算进程: {proc_result.stdout.strip()}")
            
        except Exception as e:
            print(f"GPU {gpu_id} 监控错误: {e}")
        
        time.sleep(interval)
    
    print(f"GPU {gpu_id} 监控完成: 最大利用率 {max_util:.1f}%, 最大显存使用 {max_mem}MB")

def detect_gpu_count():
    """检测系统中可用的NVIDIA GPU数量"""
    try:
        result = subprocess.run(['nvidia-smi', '-L'], capture_output=True, text=True, check=True)
        gpu_lines = [line for line in result.stdout.strip().split('\n') if line.startswith('GPU')]
        gpu_count = len(gpu_lines)
        print(f"检测到 {gpu_count} 个NVIDIA GPU:")
        for line in gpu_lines:
            print(f"  {line}")
        return gpu_count
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        print(f"无法检测GPU信息: {e}")
        print("将使用CPU渲染")
        return 0

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

# 命令行参数解析
parser = argparse.ArgumentParser()
# 数据集根目录
parser.add_argument('--data_dir', type=str, default='G:/Diffusion_Suction_DataSet', help='数据集根目录')
# 循环编号
parser.add_argument('--cycle_list', type=str, required=True, 
                   help='循环编号，支持格式: "5"(单个), "[1,10]"(区间), "{1,3,5}"(列表)')
# 场景编号  
parser.add_argument('--scene_list', type=str, required=True, 
                   help='场景编号，支持格式: "5"(单个), "[1,10]"(区间), "{1,3,5}"(列表)')
parser.add_argument('--camera_info_file', type=str, default='camera_info.yaml', help='相机参数配置文件路径')
# 是否启用GPU加速渲染
parser.add_argument('--use_gpu', action='store_true', help='设置该参数则启用GPU加速渲染')
# GPU编号（用于多进程渲染时指定特定GPU）
parser.add_argument('--gpu_id', type=int, default=0, help='指定使用的GPU编号（多进程渲染时使用）')
# 是否启用多进程并行渲染
parser.add_argument('--parallel', action='store_true', help='启用多进程并行渲染，自动检测GPU数量并分配任务')
FLAGS = parser.parse_args()

# 解析循环编号和场景编号
try:
    CYCLE_idx_list = parse_range_or_single(FLAGS.cycle_list)
    SCENE_idx_list = parse_range_or_single(FLAGS.scene_list)
except ValueError as e:
    print(f"参数解析错误: {e}")
    sys.exit(1)
print("CYCLE_idx_list")
print(CYCLE_idx_list)
print("SCENE_idx_list")
print(SCENE_idx_list)

# 获取数据集根目录
FILE_DIR = FLAGS.data_dir

# # w10 可视化时候需要多加一句
# FILE_DIR = os.path.dirname(FILE_DIR)

import bpy
import csv
import json
import numpy as np
import shutil
from math import radians
import math
import csv
# 导入相机参数工具



class BlenderVersionCompat:
    """Blender 版本兼容性辅助类"""
    
    def __init__(self):
        self.version = bpy.app.version
        self.is_new_version = self.version >= (2, 80, 0)
        print(f"检测到 Blender 版本: {self.version[0]}.{self.version[1]}.{self.version[2]}")
        
    def select_object(self, obj, state=True):
        """兼容不同版本的对象选择方法"""
        if self.is_new_version:
            obj.select_set(state)
        else:
            obj.select = state
            
    def get_light_type(self):
        """获取灯光类型名称"""
        return 'LIGHT' if self.is_new_version else 'LAMP'
        
    def get_light_add_op(self):
        """获取添加灯光的操作符"""
        return bpy.ops.object.light_add if self.is_new_version else bpy.ops.object.lamp_add
    
    def import_obj_file(self, filepath):
        """兼容不同版本的OBJ文件导入方法"""
        try:
            # 尝试新版本的导入方式 (Blender 3.0+)
            if hasattr(bpy.ops.wm, 'obj_import'):
                bpy.ops.wm.obj_import(filepath=filepath)
                #print("使用新版本 OBJ 导入 API (3.0+)")
            # 尝试中等版本的导入方式 (Blender 2.80-2.93)
            elif hasattr(bpy.ops, 'import_scene') and hasattr(bpy.ops.import_scene, 'obj'):
                bpy.ops.import_scene.obj(filepath=filepath)
                #print("使用中版本 OBJ 导入 API (2.80-2.93)")
            # 旧版本的导入方式 (Blender 2.79-)
            else:
                bpy.ops.import_scene.obj(filepath=filepath)
                #print("使用旧版本 OBJ 导入 API (2.79-)")
        except Exception as e:
            print(f"OBJ 导入失败，尝试其他方法: {e}")
            # 备用方法：如果上述都失败，尝试直接调用
            try:
                bpy.ops.wm.obj_import(filepath=filepath)
            except:
                try:
                    bpy.ops.import_scene.obj(filepath=filepath)
                except Exception as final_e:
                    raise Exception(f"无法导入 OBJ 文件 {filepath}: {final_e}")

# 创建全局兼容性对象
blender_compat = BlenderVersionCompat()

OBJ_PATH =  os.path.join(FILE_DIR, 'OBJ')
OUTDIR_physics_result_dir =  os.path.join(FILE_DIR, 'physics_result')
OUTDIR_dir_depth_images =  os.path.join(FILE_DIR, 'depth_images')
if not os.path.exists(OUTDIR_dir_depth_images):
    os.makedirs(OUTDIR_dir_depth_images)
OUTDIR_dir_segment_images =  os.path.join(FILE_DIR, 'segment_images')
if not os.path.exists(OUTDIR_dir_segment_images):
    os.makedirs(OUTDIR_dir_segment_images)
OUTDIR_dir_rgb_images =  os.path.join(FILE_DIR, 'rgb_images')
if not os.path.exists(OUTDIR_dir_rgb_images):
    os.makedirs(OUTDIR_dir_rgb_images)

class BlenderRenderClass:
    def __init__(self):
        camera_info_path = FLAGS.camera_info_file
        if not os.path.isabs(camera_info_path):
            camera_info_path = os.path.join(os.path.dirname(__file__), '..', 'config', camera_info_path)
        camera_info_path = os.path.abspath(camera_info_path)
        print(f"加载相机参数文件: {camera_info_path}")
        self.cam_info = CameraInfo(camera_info_path)
        depth_graph_divide =  2    # 深度图缩放因子
        depth_graph_less = 3       # 深度图阈值

        self.CAMERA_FOCAL_LEN = self.cam_info.focal_length
        self.CAMERA_SENSOR_SIZE = self.cam_info.sensor_size
        self.CAMERA_LOCATION = self.cam_info.cam_translation_vector
        self.CAMERA_ROTATION = self.cam_info.cam_quaternions
        self.img_w = self.cam_info.intrinsic_matrix[0,2] * 2
        self.img_h = self.cam_info.intrinsic_matrix[1,2] * 2
        self.CAMERA_RESOLUTION = [int(self.img_w), int(self.img_h)]
        self.DEPTH_DIVIDE = depth_graph_divide
        self.DEPTH_LESS = depth_graph_less
        unit_of_obj = 'mm'
        if unit_of_obj == 'mm':
            self.meshScale = [0.001, 0.001, 0.001]  # 毫米转米
        elif unit_of_obj == 'm':
            self.meshScale = [1, 1, 1]

    def set_device(self):
        if FLAGS.use_gpu:
            # # 设置CUDA可见设备（用于多进程渲染）
            # if hasattr(FLAGS, 'gpu_id') and FLAGS.gpu_id is not None:
            #     os.environ['CUDA_VISIBLE_DEVICES'] = str(FLAGS.gpu_id)
            #     print(f'设置CUDA_VISIBLE_DEVICES为GPU {FLAGS.gpu_id}')
            
            bpy.context.scene.cycles.device = 'GPU'
            prefs = bpy.context.preferences.addons['cycles'].preferences
            prefs.compute_device_type = 'CUDA'
            prefs.get_devices()
            
            gpu_found = False
            print(f'检测到 {len(prefs.devices)} 个计算设备')
            print(f'当前CUDA_VISIBLE_DEVICES: {os.environ.get("CUDA_VISIBLE_DEVICES", "未设置")}')
            
            # 统计各类型设备数量
            device_types = {}
            cuda_devices = []
            optix_devices = []
            cpu_devices = []
            
            # 第一步：收集所有设备信息
            for i, device in enumerate(prefs.devices):
                device_type = device.type
                if device_type not in device_types:
                    device_types[device_type] = 0
                device_types[device_type] += 1
                
                print(f'设备 {i}: 类型={device.type}, 名称={device.name}')
                
                if device.type == 'CUDA':
                    cuda_devices.append(i)
                elif device.type == 'OPTIX':
                    optix_devices.append(i)
                elif device.type == 'CPU':
                    cpu_devices.append(i)
            
            # 第二步：设备启用策略
            # 优先启用OPTIX设备，如果没有则启用CUDA设备
            enabled_devices = []
            
            if optix_devices:
                # 如果有OPTIX设备，启用所有OPTIX设备
                for i in optix_devices:
                    prefs.devices[i].use = True
                    enabled_devices.append(i)
                    gpu_found = True
                    print(f'已启用OPTIX GPU (设备索引: {i})')
                
                # 禁用对应的CUDA设备（避免重复）
                for i in cuda_devices:
                    prefs.devices[i].use = False
                    print(f'禁用对应的CUDA设备 (设备索引: {i})')
                    
            elif cuda_devices:
                # 如果没有OPTIX但有CUDA设备，启用所有CUDA设备
                for i in cuda_devices:
                    prefs.devices[i].use = True
                    enabled_devices.append(i)
                    gpu_found = True
                    print(f'已启用CUDA GPU (设备索引: {i})')
            
            # 禁用CPU设备（通常不需要CPU和GPU同时渲染）
            for i in cpu_devices:
                prefs.devices[i].use = False
                print(f'禁用CPU设备 (设备索引: {i})')
            
            # 打印设备统计信息
            print(f'设备类型统计: {device_types}')
            print(f'CUDA设备索引: {cuda_devices}')
            print(f'OPTIX设备索引: {optix_devices}')
            print(f'CPU设备索引: {cpu_devices}')
            print(f'已启用的GPU设备: {enabled_devices}')
            
            # 验证GPU是否真的被启用
            if enabled_devices:
                print("验证GPU启用状态:")
                for i, device in enumerate(prefs.devices):
                    if device.use:
                        print(f"  设备 {i} ({device.type}): {device.name} - 已启用")
            
            if not gpu_found:
                print('警告: 未找到可用的GPU，切换到CPU渲染')
                bpy.context.scene.cycles.device = 'CPU'
        else:
            bpy.context.scene.cycles.device = 'CPU'
            print('已设置为CPU渲染')

    def camera_set(self):
        # 设置渲染引擎为CYCLES
        bpy.data.scenes["Scene"].render.engine = "CYCLES"

        # 设置相机内参
        bpy.data.scenes["Scene"].render.resolution_x = self.CAMERA_RESOLUTION[0]
        bpy.data.scenes["Scene"].render.resolution_y = self.CAMERA_RESOLUTION[1]

        bpy.data.scenes["Scene"].render.resolution_percentage = 100

        # 设置相机焦距和传感器尺寸, 单位为毫米
        bpy.data.cameras["Camera"].type = "PERSP"
        bpy.data.cameras["Camera"].lens = self.CAMERA_FOCAL_LEN
        bpy.data.cameras["Camera"].lens_unit = "MILLIMETERS"
        bpy.data.cameras["Camera"].sensor_width = self.CAMERA_SENSOR_SIZE[0]
        bpy.data.cameras["Camera"].sensor_height = self.CAMERA_SENSOR_SIZE[1]
        
        # 传感器适配方式为宽度适配
        bpy.data.cameras["Camera"].sensor_fit = "HORIZONTAL"
        
        # 设置像素长宽比
        bpy.data.scenes["Scene"].render.pixel_aspect_x = 1.0
   
        bpy.data.scenes["Scene"].render.pixel_aspect_y = self.CAMERA_SENSOR_SIZE[1] * self.CAMERA_RESOLUTION[0] / \
                                                         self.CAMERA_RESOLUTION[1] / self.CAMERA_SENSOR_SIZE[0]
        bpy.data.scenes["Scene"].cycles.progressive = "BRANCHED_PATH"
        bpy.data.scenes["Scene"].cycles.aa_samples = 1
        bpy.data.scenes["Scene"].cycles.preview_aa_samples = 1
       
        bpy.data.objects["Camera"].location = [self.cam_info.cam_translation_vector[0],
                                               self.cam_info.cam_translation_vector[1],
                                               self.cam_info.cam_translation_vector[2]]
        bpy.data.objects["Camera"].rotation_mode = 'QUATERNION'
        # 设置相机的四元数旋转, 注意要把四元数的顺序调整为 [qw, qx, qy, qz]
        bpy.data.objects["Camera"].rotation_quaternion = [self.cam_info.cam_quaternions[3],
                                                          self.cam_info.cam_quaternions[0],
                                                          self.cam_info.cam_quaternions[1],
                                                          self.cam_info.cam_quaternions[2]]
        # 让相机坐标系绕X轴旋转180度, 适配Blender坐标系
        bpy.data.objects["Camera"].rotation_mode = 'XYZ'
        # bpy.data.objects["Camera"].rotation_euler[0] = bpy.data.objects["Camera"].rotation_euler[0] + math.pi

        # 获取 Blender 版本并设置兼容性参数
        light_type = blender_compat.get_light_type()
        light_add_op = blender_compat.get_light_add_op()
        print(f"使用 {'新版本' if blender_compat.is_new_version else '旧版本'} 灯光 API")

        # 清除现有的灯光对象, 避免多余光源影响渲染
        bpy.ops.object.select_all(action='DESELECT')
        bpy.ops.object.select_by_type(type=light_type)  # 选择所有灯光对象
        bpy.ops.object.delete()

        # 创建平行光(Sun Light), 用于模拟环境主光源
        light_add_op(type='SUN', location=(0, 0, 2))  # 添加平行光源并设置其位置
        sun_light = bpy.context.object  # 获取刚创建的光源对象
        sun_light.name = "Sun_Light"  # 给光源命名
        # 设置平行光的属性
        sun_light.data.energy = 10  # 设置光源强度
        sun_light.data.color = (1, 1, 1)  # 设置光源颜色为白色 (RGB)
        sun_light.data.use_nodes = True  # 启用节点系统(如果需要控制光源的其他属性)
        
        # 创建多个点光源, 增强场景整体照明
        locations_z = 1.3
        locations = [[0,0,locations_z],[0,locations_z*0.5,locations_z],[locations_z*0.5,0,locations_z],[0,-locations_z*0.5,locations_z],[-locations_z*0.5,0,locations_z]]
        for i in range(5):
            light_add_op(type='POINT', location=locations[i])
            point_light = bpy.context.object
            point_light.name = "Point_Light"
            point_light.data.energy = 10  # 设置光源强度
            
    def read_csv(self, csv_path):      
        # 读取csv文件, 返回物体名称、位姿、索引
        with open(csv_path,'r') as csv_file:  
            all_lines=csv.reader(csv_file) 
            list_file = [i for i in all_lines]  # 读取所有行
        array_file = np.array(list_file)[1:]
        obj_name = array_file[:,0]
        obj_index = array_file[:,1].astype('int')
        pose = array_file[:,2:9].astype('float32')
        return obj_name, pose, obj_index

    def import_obj(self, obj_name, pose, instance_index):
        # 导入指定物体到Blender场景中, 并设置其位姿
        
        # 删除场景中所有网格对象
        for o in bpy.data.objects:
            if o.type == 'MESH':
                blender_compat.select_object(o, True)
            else:
                blender_compat.select_object(o, False)
        bpy.ops.object.delete()
    
        for instance_index_ in instance_index:
            file_path = os.path.join(OBJ_PATH, obj_name[instance_index_] ,'object.obj')
            print(f"正在导入 OBJ 文件: {file_path}")
            
            # 使用兼容性导入方法
            blender_compat.import_obj_file(file_path)
            
            # 获取刚导入的物体对象（假定每次只导入一个物体，且为选中状态）
            instance = bpy.context.selected_objects[0]
            print(bpy.context.selected_objects)  # 打印当前选中的物体列表，便于调试
            print(instance_index_)  # 打印当前物体的分割索引，便于调试
            # 设置物体的 pass_index，用于后续分割图中唯一标识（ObjectInfo节点会用到）
            instance.pass_index = instance_index_
            # 设置物体的缩放比例（毫米转米，Blender内部单位为米）
            instance.scale = [0.001, 0.001, 0.001]
            # 设置物体的位置（x, y, z），从 csv pose 数据读取
            instance.location = [pose[instance_index_][0], pose[instance_index_][1], pose[instance_index_][2]]
            # 设置物体的旋转模式为四元数，便于精确控制三维旋转
            instance.rotation_mode = 'QUATERNION'
            # 设置物体的四元数旋转（qw, qx, qy, qz），从 csv pose 数据读取
            instance.rotation_quaternion = [pose[instance_index_][3], pose[instance_index_][4], pose[instance_index_][5], pose[instance_index_][6]]

    def grb_graph(self, rgb_scene_path):
        # 使用节点合成系统输出RGB图像
        bpy.data.scenes["Scene"].use_nodes = 1

        # 定义合成节点
        scene = bpy.context.scene
        nodes = scene.node_tree.nodes
        links = scene.node_tree.links
        for node in nodes:
            nodes.remove(node)
        
        output_file_rgb = nodes.new("CompositorNodeOutputFile")
        output_file_rgb.base_path = rgb_scene_path
        output_file_rgb.format.file_format = "OPEN_EXR"
        output_file_rgb.format.color_mode = "RGB"
        output_file_rgb.format.color_depth = '32'
        render_layers = nodes.new("CompositorNodeRLayers")
        links.new(render_layers.outputs['Image'], output_file_rgb.inputs['Image'])

    # 使用节点配置深度图和分割图的输出
    def depth_graph(self, depth_path, segment_path):
        # 启用节点合成功能
        bpy.data.scenes["Scene"].use_nodes = 1
        
        # 首先确保启用深度输出
        scene = bpy.context.scene
        
        # 兼容不同版本的视图层访问方式
        view_layer = None
        try:
            # 尝试新版本的方式 (Blender 2.80+)
            if hasattr(scene, 'view_layers') and hasattr(scene.view_layers, 'active'):
                view_layer = scene.view_layers.active
                print("使用新版本视图层访问方式 (view_layers.active)")
            elif hasattr(scene, 'view_layers') and len(scene.view_layers) > 0:
                view_layer = scene.view_layers[0]
                print("使用新版本视图层访问方式 (view_layers[0])")
            # 尝试旧版本的方式 (Blender 2.79-)
            elif hasattr(scene, 'layers'):
                # 旧版本使用 render layers 而不是 view layers
                print("检测到旧版本 Blender，使用传统渲染层设置")
                # 对于旧版本，直接在场景上设置
                if hasattr(scene.render, 'use_pass_z'):
                    scene.render.use_pass_z = True
                    print("已启用深度通道 (scene.render.use_pass_z)")
                view_layer = None  # 旧版本不需要view_layer
            else:
                # 最后的尝试：直接使用当前视图层
                view_layer = bpy.context.view_layer
                print("使用当前视图层 (bpy.context.view_layer)")
        except Exception as e:
            print(f"获取视图层时出错: {e}")
            # 备用方案：尝试直接使用上下文
            try:
                view_layer = bpy.context.view_layer
                print("使用备用方案: bpy.context.view_layer")
            except:
                print("警告: 无法获取视图层，将跳过深度通道设置")
                view_layer = None
        
        # 启用深度通道 - 这是关键步骤
        if view_layer is not None:
            if hasattr(view_layer, 'use_pass_z'):
                view_layer.use_pass_z = True
                print("已启用深度通道 (use_pass_z)")
            
            # 对于某些Blender版本，可能需要启用其他深度相关设置
            if hasattr(view_layer, 'use_pass_depth'):
                view_layer.use_pass_depth = True
                print("已启用深度通道 (use_pass_depth)")
        else:
            print("无法获取视图层，尝试在场景级别设置深度通道")
            # 尝试在场景级别设置
            if hasattr(scene.render, 'use_pass_z'):
                scene.render.use_pass_z = True
                print("已在场景级别启用深度通道")

        # 定义合成节点
        nodes = scene.node_tree.nodes
        links = scene.node_tree.links
        for node in nodes:
            nodes.remove(node)

        render_layers = nodes.new("CompositorNodeRLayers")
        
        # 等待一点时间让深度通道设置生效，然后获取深度输出
        print("正在获取深度输出...")
        
        # 列出所有可用的输出进行调试
        available_outputs = [output.name for output in render_layers.outputs]
        print(f"可用的渲染层输出: {available_outputs}")
        
        # 使用更安全的方法查找深度输出
        depth_output = None
        depth_output_name = None
        
        # 尝试多种可能的深度输出名称
        possible_depth_names = ['Depth', 'Z']
        
        for candidate_name in possible_depth_names:
            print(f"尝试查找深度输出: {candidate_name}")
            
            # 检查是否在可用输出列表中
            if candidate_name in available_outputs:
                # 通过遍历的方式安全访问
                for output in render_layers.outputs:
                    if output.name == candidate_name:
                        depth_output = output
                        depth_output_name = candidate_name
                        print(f"成功找到深度输出: {candidate_name}")
                        break
                        
                if depth_output is not None:
                    break
                else:
                    print(f"虽然 {candidate_name} 在列表中，但无法访问")
            else:
                print(f"{candidate_name} 不在可用输出列表中")
        
        # 如果标准方法失败，尝试遍历所有输出
        if depth_output is None:
            print("标准方法失败，遍历所有输出寻找深度相关的...")
            for i, output in enumerate(render_layers.outputs):
                output_name = output.name
                print(f"检查输出 {i}: '{output_name}'")
                
                # 检查是否是深度相关的输出
                if output_name.lower() in ['depth', 'z'] or 'depth' in output_name.lower():
                    depth_output = output
                    depth_output_name = output_name
                    print(f"找到深度相关输出: {output_name}")
                    break
        
        # 如果还是找不到，这可能意味着需要强制更新渲染层
        if depth_output is None:
            print("仍然找不到深度输出，尝试强制更新渲染层...")
            
            # 强制更新场景
            bpy.context.view_layer.update()
            
            # 重新创建渲染层节点
            nodes.remove(render_layers)
            render_layers = nodes.new("CompositorNodeRLayers")
            
            # 再次尝试
            updated_outputs = [output.name for output in render_layers.outputs]
            print(f"更新后的输出: {updated_outputs}")
            
            for output in render_layers.outputs:
                if output.name in ['Depth', 'Z']:
                    depth_output = output
                    depth_output_name = output.name
                    print(f"强制更新后找到深度输出: {output.name}")
                    break
        
        # 最终检查
        if depth_output is None:
            error_msg = f"无法找到深度输出。可用输出: {available_outputs}。请确保启用了深度通道。"
            print(f"错误: {error_msg}")
            raise Exception(error_msg)
        
        print(f"最终使用的深度输出: {depth_output_name}")
        
        divide = nodes.new("CompositorNodeMath")
        divide.operation = "DIVIDE"
        divide.inputs[1].default_value = self.DEPTH_DIVIDE
        less_than = nodes.new("CompositorNodeMath")
        less_than.operation = "LESS_THAN"
        less_than.inputs[1].default_value = self.DEPTH_LESS
        multiply = nodes.new("CompositorNodeMath")
        multiply.operation = "MULTIPLY"

        # 一个输出用于深度图, 另一个用于标签图
        output_file_depth = nodes.new("CompositorNodeOutputFile")
        output_file_depth.base_path = depth_path
        output_file_depth.format.file_format = "PNG"
        output_file_depth.format.color_mode = "BW"
        output_file_depth.format.color_depth = '16'

        output_file_label = nodes.new("CompositorNodeOutputFile")
        output_file_label.base_path = segment_path
        output_file_label.format.file_format = "OPEN_EXR"
        output_file_label.format.color_mode = "RGB"
        output_file_label.format.color_depth = '32'

        composite = nodes.new("CompositorNodeComposite")
        viewer = nodes.new("CompositorNodeViewer")

        links.new(render_layers.outputs['Image'], composite.inputs['Image'])
        links.new(depth_output, less_than.inputs[0])  # 使用找到的深度输出
        links.new(depth_output, multiply.inputs[0])   # 使用找到的深度输出

        links.new(less_than.outputs[0], multiply.inputs[1])
        links.new(multiply.outputs[0], divide.inputs[0])

        links.new(divide.outputs[0], output_file_depth.inputs['Image'])
        links.new(divide.outputs[0], viewer.inputs['Image'])
        links.new(render_layers.outputs['Image'], output_file_label.inputs['Image'])

    # 定义物体的材质(如颜色), 并让所有物体指向同一个材质
    def label_graph(self, label_number):

        # 遍历场景中的所有物体
        for obj in bpy.context.scene.objects:
            # 只处理网格对象(你可以根据需要调整条件)
            if obj.type == 'MESH':
                # 确保物体有材质槽
                if obj.data.materials:
                    # 清空物体的所有材质槽
                    obj.data.materials.clear()
                    print("delet object  materials")

        mymat = bpy.data.materials.get('mymat')
        if not mymat:
            mymat = bpy.data.materials.new('mymat')
            mymat.use_nodes = True

        # 删除初始节点
        nodes = mymat.node_tree.nodes
        links = mymat.node_tree.links
        for node in nodes:
            nodes.remove(node)

        # 配置颜色渐变节点
        ColorRamp = nodes.new(type="ShaderNodeValToRGB")
        ColorRamp.color_ramp.interpolation = 'LINEAR'
        ColorRamp.color_ramp.color_mode = 'RGB'

        ColorRamp.color_ramp.elements[0].color[:3] = [1.0, 0.0, 0.0]  # 红色
        ColorRamp.color_ramp.elements[1].color[:3] = [1.0, 1.0, 0.0]  # 黄色

        # 根据物体数量添加分段
        # 创建ObjectInfo节点，用于获取每个物体的属性（如Index、Random等）
        ObjectInfo = nodes.new(type="ShaderNodeObjectInfo")
        # 创建材质输出节点
        OutputMat = nodes.new(type="ShaderNodeOutputMaterial")
        # 创建发射材质节点，使物体表面显示为纯色（不受光照影响，适合分割标签）
        Emission = nodes.new(type="ShaderNodeEmission")

        # 创建数学节点，用于将物体的Index归一化到0~1区间
        Math = nodes.new(type="ShaderNodeMath")
        Math.operation = "DIVIDE"
        Math.inputs[1].default_value = label_number

        # 连接ObjectInfo的Object Index输出（outputs[3]）到Math节点，实现分割标签的唯一性
        # 注意此处的outputs[3]是Object Index（pass_index），可能因Blender版本不同而有所变化
        # 建议运行前取消下面代码的注释，先打印ObjectInfo.outputs的名称，确认索引是否正确
        # print("ObjectInfo节点输出顺序：")
        # for i, out in enumerate(ObjectInfo.outputs):
        #     print(f"{i}: {out.name}")
        links.new(ObjectInfo.outputs[3], Math.inputs[0])  # Object Index（pass_index）/最大值
        # 连接归一化后的Index到ColorRamp，实现分段或渐变颜色映射
        links.new(Math.outputs[0], ColorRamp.inputs[0])
        # 连接ColorRamp输出到Emission，使物体表面显示为分割色
        links.new(ColorRamp.outputs[0], Emission.inputs[0])
        # 连接Emission到材质输出，最终决定物体表面颜色
        links.new(Emission.outputs[0], OutputMat.inputs[0])

        # 让所有网格对象都使用同一个材质
        objects = bpy.data.objects
        count = 0
        for obj in objects:
            if obj.type == 'MESH':
                count+=1
                if not 'mymat' in obj.data.materials:
                    obj.data.materials.append(mymat)

    def render_scenes(self):
        times = []     
        self.set_device()  # 设置渲染设备为GPU或CPU
        self.camera_set()  # 设置相机和光源
        for cycle_id in CYCLE_idx_list:
            for scene_id in SCENE_idx_list:
                start_time = time.time()  # 记录起始时间戳
                # 获取物体名称列表和位姿数组(x, y, z, qw, qx, qy, qz)
        
                csv_path = os.path.join(OUTDIR_physics_result_dir, 'cycle_{:0>4}'.format(cycle_id),"{:0>3}".format(scene_id), "{:0>3}.csv".format(scene_id))
                obj_name, pose, segment_index = self.read_csv(csv_path)

                self.import_obj(obj_name, pose, segment_index)  # 导入所有物体并设置位姿

                depth_scene_path = os.path.join(OUTDIR_dir_depth_images,'cycle_{:0>4}'.format(cycle_id),"{:0>3}".format(scene_id))
                segment_scene_path = os.path.join(OUTDIR_dir_segment_images, 'cycle_{:0>4}'.format(cycle_id),"{:0>3}".format(scene_id))
                rgb_scene_path = os.path.join(OUTDIR_dir_rgb_images, 'cycle_{:0>4}'.format(cycle_id),"{:0>3}".format(scene_id))
                if not os.path.exists(depth_scene_path):
                    os.makedirs(depth_scene_path)
                if not os.path.exists(segment_scene_path):
                    os.makedirs(segment_scene_path)
                if not os.path.exists(rgb_scene_path):
                    os.makedirs(rgb_scene_path)

                print(f"正在渲染 Cycle: {cycle_id:04d}, Scene: {scene_id:03d}")
                
                self.grb_graph(rgb_scene_path)  # 配置RGB图输出节点
                bpy.ops.render.render()         # 渲染并输出RGB图
                self.depth_graph(depth_scene_path, segment_scene_path)  # 配置深度图和分割图输出节点
                self.label_graph(len(obj_name) - 1)  # 配置分割标签材质
                bpy.ops.render.render()         # 渲染并输出深度图和分割图
                times.append(time.time()-start_time)

                print(f"完成渲染 Cycle: {cycle_id:04d}, Scene: {scene_id:03d}, 耗时: {times[-1]:.2f}秒")
                # 主动清理未使用的数据块和垃圾回收, 防止内存不够用
                bpy.ops.outliner.orphans_purge(do_recursive=True)
                gc.collect()
                
        np.save('times.npy', times)
        print(f"总计渲染 {len(times)} 个场景，平均耗时: {np.mean(times):.2f}秒")
        print(times)

def render_worker(gpu_id, cycle_scene_pairs, data_dir, camera_info_file):
    """工作进程函数，每个进程使用指定的GPU"""
    print(f"GPU {gpu_id} 工作进程启动，处理 {len(cycle_scene_pairs)} 个任务")
    
    # 设置CUDA可见设备为单个GPU（用于多进程时的GPU隔离）
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
    print(f"进程GPU {gpu_id}: 设置CUDA_VISIBLE_DEVICES={gpu_id}")
    
    # 启动GPU监控线程
    monitor_thread = threading.Thread(target=monitor_gpu_usage, args=(gpu_id, 300, 2))
    monitor_thread.daemon = True
    monitor_thread.start()
    
    # 构造命令行参数，为每个cycle-scene对分别执行
    for cycle_id, scene_id in cycle_scene_pairs:
        cmd = [
            'blender', '--background', '--python', __file__,
            '--', 
            '--data_dir', data_dir,
            '--cycle_list', str(cycle_id),
            '--scene_list', str(scene_id),
            '--camera_info_file', camera_info_file,
            '--gpu_id', str(gpu_id),
            '--use_gpu'
        ]
        
        print(f"GPU {gpu_id} 开始渲染: Cycle {cycle_id:04d}, Scene {scene_id:03d}")
        
        # 在渲染前检查GPU状态
        try:
            gpu_status_cmd = ['nvidia-smi', '--query-gpu=name,memory.used,memory.total', 
                             '--format=csv,noheader', '-i', str(gpu_id)]
            gpu_status = subprocess.run(gpu_status_cmd, capture_output=True, text=True, check=True)
            print(f"GPU {gpu_id} 渲染前状态: {gpu_status.stdout.strip()}")
        except:
            pass
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            print(f"GPU {gpu_id} 完成渲染: Cycle {cycle_id:04d}, Scene {scene_id:03d}")
        except subprocess.CalledProcessError as e:
            print(f"GPU {gpu_id} 渲染失败: Cycle {cycle_id:04d}, Scene {scene_id:03d}")
            print(f"错误信息: {e.stderr}")
    
    print(f"GPU {gpu_id} 工作进程完成")

def main_parallel():
    """主函数：自动检测GPU并分配任务到多个GPU进行并行渲染"""
    print("=" * 50)
    print("启动多进程并行渲染模式")
    print("=" * 50)
    
    # 检测可用GPU数量
    gpu_count = detect_gpu_count()
    
    if gpu_count == 0:
        print("未检测到GPU，退出并行模式")
        return
    
    # 获取实际的GPU编号列表
    cuda_visible_devices = os.environ.get('CUDA_VISIBLE_DEVICES', None)
    if cuda_visible_devices:
        # 如果设置了CUDA_VISIBLE_DEVICES，使用其中指定的GPU编号
        available_gpus = [int(x.strip()) for x in cuda_visible_devices.split(',')]
        print(f"从CUDA_VISIBLE_DEVICES获取GPU编号: {available_gpus}")
    else:
        # 如果没有设置，使用0到gpu_count-1
        available_gpus = list(range(gpu_count))
        print(f"使用默认GPU编号: {available_gpus}")
    
    # 确保GPU数量一致
    if len(available_gpus) != gpu_count:
        print(f"警告: CUDA_VISIBLE_DEVICES中的GPU数量({len(available_gpus)})与nvidia-smi检测到的GPU数量({gpu_count})不一致")
        # 使用较小的数量
        gpu_count = min(len(available_gpus), gpu_count)
        available_gpus = available_gpus[:gpu_count]
        print(f"使用GPU编号: {available_gpus}")
    
    # 解析循环编号和场景编号
    try:
        cycle_list = parse_range_or_single(FLAGS.cycle_list)
        scene_list = parse_range_or_single(FLAGS.scene_list)
    except ValueError as e:
        print(f"参数解析错误: {e}")
        return
    
    # 生成所有cycle和scene的组合
    all_pairs = []
    for cycle_id in cycle_list:
        for scene_id in scene_list:
            all_pairs.append((cycle_id, scene_id))
    
    total_tasks = len(all_pairs)
    print(f"总任务数: {total_tasks}")
    print(f"可用GPU数: {gpu_count}")
    print(f"GPU编号列表: {available_gpus}")
    
    # 将任务分配到各个GPU
    chunk_size = total_tasks // gpu_count
    remainder = total_tasks % gpu_count
    

    chunks = []
    start_idx = 0
    
    for i in range(gpu_count):
        # 为前remainder个GPU分配额外的一个任务
        current_chunk_size = chunk_size + (1 if i < remainder else 0)
        end_idx = start_idx + current_chunk_size
        
        if start_idx < total_tasks:
            chunk = all_pairs[start_idx:end_idx]
            chunks.append((available_gpus[i], chunk))  # 使用实际的GPU编号
            print(f"GPU {available_gpus[i]}: 分配 {len(chunk)} 个任务 (任务索引: {start_idx}-{end_idx-1})")
        else:
            chunks.append((available_gpus[i], []))
            print(f"GPU {available_gpus[i]}: 无任务分配")
        
        start_idx = end_idx
    
    # 启动多个进程
    start_time = time.time()
    processes = []
    
    for gpu_id, chunk in chunks:
        if chunk:  # 确保有任务要处理
            p = mp.Process(target=render_worker, 
                          args=(gpu_id, chunk, FLAGS.data_dir, FLAGS.camera_info_file))
            p.start()
            processes.append(p)
    
    print(f"\n已启动 {len(processes)} 个渲染进程")
    print("等待所有进程完成...")
    
    # 等待所有进程完成
    for i, p in enumerate(processes):
        p.join()
        print(f"进程 {i} 已完成")
    
    total_time = time.time() - start_time
    print("=" * 50)
    print("所有GPU渲染完成!")
    print(f"总耗时: {total_time:.2f}秒")
    print(f"平均每个任务: {total_time/total_tasks:.2f}秒")
    print("=" * 50)

if __name__ == '__main__':
    import time
    
    # 检查是否启用并行模式
    if FLAGS.parallel:
        main_parallel()
    else:
        # 单进程模式（原有逻辑）
        blender_generator = BlenderRenderClass()
        blender_generator.render_scenes()

    

