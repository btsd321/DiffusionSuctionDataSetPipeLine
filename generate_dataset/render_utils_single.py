# -*- coding:utf-8 -*-
"""
本文件用于在Blender中对单个物体进行批量渲染, 自动导入物体模型、设置相机参数, 并输出分割标签图。适用于数据集单物体分割标签的自动生成与渲染流程。

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
from multiprocessing import Pool, cpu_count
from functools import partial
import time

def parse_range_or_single(input_str):
    """
    解析输入字符串，支持以下格式：
    - 单个值: "5" -> [5]
    - 区间: "[1,10]" -> [1,2,3,4,5,6,7,8,9,10]
    - 列表: "{1,3,5}" -> [1,3,5]
    """
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
parser.add_argument('--cycle_list', type=str, default='1', help='循环编号，支持格式: "5"(单个), "[1,10]"(区间), "{1,3,5}"(列表)')
parser.add_argument('--scene_list', type=str, default='1', help='场景编号，支持格式: "5"(单个), "[1,10]"(区间), "{1,3,5}"(列表)')
parser.add_argument('--camera_info_file', type=str, default='camera_info.yaml', help='相机参数配置文件路径')
# 是否启用GPU加速渲染
parser.add_argument('--use_gpu', action='store_true', help='设置该参数则启用GPU加速渲染')
parser.add_argument('--save_img_type', type=str, default='png', choices=['exr', 'png'], 
                    help='保存分割图像的格式: exr(高精度,慢) 或 png(快速mask,推荐)')
parser.add_argument('--fast_mode', action='store_true', 
                    help='启用快速模式：降低分辨率、减少采样等，用于快速预览')
parser.add_argument('--disable_print', action='store_true', 
                    help='禁用详细打印输出以提高性能')
parser.add_argument('--headless', action='store_true', 
                    help='强制无头渲染模式，避免OpenGL上下文问题（适用于WSL）')
parser.add_argument('--ultra_fast', action='store_true', 
                    help='极速模式：最大化性能优化，适用于batch mask生成')
parser.add_argument('--num_workers', type=int, default=1, 
                    help='并行渲染的进程数量，默认为1（串行）。建议不超过CPU核心数的一半')
parser.add_argument('--input_type', type=str, default='continuous_scences', choices=['discrete_scences', 'continuous_scences'],
                    help='输入类型，discrete_scences表示输入为离散场景此时忽略cycle_list参数和scene_list参数，continuous_scences表示连续场景此时cycle_list参数和scene_list参数为循环次数和场景数量')
FLAGS = parser.parse_args()

# 失败的循环-场景列表（用于离散模式）
failed_cycles_scenes = [
    # [1, 34],
    [3, 15],
    [4, 35],
    [4, 46],
    [5, 33],
    [6, 18],
    [6, 40],
    [6, 43],
    [6, 50],
    [7, 32],
    [8, 16],
    [8, 34],
    [8, 38],
    [8, 41],
    [9, 47],
    [10, 23],
    [10, 49]
]

def generate_cycle_scene_lists(failed_list):
    """
    从失败的循环-场景列表生成CYCLE_idx_list和SCENE_idx_list
    
    Args:
        failed_list (list): [[cycle, scene], ...] 格式的失败列表
        
    Returns:
        tuple: (cycle_list, scene_list) 分别对应每个失败项的循环和场景列表
    """
    if not failed_list:
        print("⚠️ 警告: failed_cycles_scenes 列表为空")
        return [], []
    
    cycle_list = []
    scene_list = []
    
    for cycle, scene in failed_list:
        cycle_list.append(cycle)
        scene_list.append(scene)
    
    print(f"📊 从失败列表生成的配对:")
    print(f"   循环数量: {len(cycle_list)} 个")
    print(f"   场景数量: {len(scene_list)} 个")
    print(f"   循环范围: {min(cycle_list)} - {max(cycle_list)}")
    print(f"   场景范围: {min(scene_list)} - {max(scene_list)}")
    
    # 显示前几个配对作为示例
    print(f"📋 前5个循环-场景配对:")
    for i, (cycle, scene) in enumerate(failed_list[:5]):
        print(f"   [{i}]: 循环{cycle} -> 场景{scene}")
    
    if len(failed_list) > 5:
        print(f"   ... 还有 {len(failed_list) - 5} 个配对")
    
    return cycle_list, scene_list

def render_single_object_task(args):
    """
    独立的单物体渲染任务，用于多进程执行
    
    Args:
        args: 包含渲染参数的元组 (cycle_id, scene_id, obj_index, FLAGS_dict)
    
    Returns:
        tuple: (success, cycle_id, scene_id, obj_index, message)
    """
    cycle_id, scene_id, obj_index, flags_dict = args
    
    try:
        # 重新设置FLAGS（因为在子进程中无法直接访问全局变量）
        class FlagsClass:
            def __init__(self, flags_dict):
                for key, value in flags_dict.items():
                    setattr(self, key, value)
        
        local_flags = FlagsClass(flags_dict)
        
        # 导入Blender相关模块（在子进程中重新导入）
        import bpy
        import csv
        import json
        import numpy as np
        import shutil
        from math import radians
        import math
        
        # 重新设置全局变量和路径
        FILE_DIR = local_flags.data_dir
        OBJ_PATH = os.path.join(FILE_DIR, 'OBJ')
        OUTDIR_physics_result_dir = os.path.join(FILE_DIR, 'physics_result')
        OUTDIR_dir_segment_images = os.path.join(FILE_DIR, 'segment_images_single')
        
        if not os.path.exists(OUTDIR_dir_segment_images):
            os.makedirs(OUTDIR_dir_segment_images, exist_ok=True)
        
        # 重新导入Blender兼容性类
        class BlenderVersionCompat:
            def __init__(self):
                self.version = bpy.app.version
                self.is_new_version = self.version >= (2, 80, 0)

            def select_object(self, obj, state=True):
                if self.is_new_version:
                    obj.select_set(state)
                else:
                    obj.select = state

            def get_light_type(self):
                return 'LIGHT' if self.is_new_version else 'LAMP'

            def get_light_add_op(self):
                return bpy.ops.object.light_add if self.is_new_version else bpy.ops.object.lamp_add

            def import_obj_file(self, filepath):
                try:
                    if hasattr(bpy.ops.wm, 'obj_import'):
                        bpy.ops.wm.obj_import(filepath=filepath)
                    elif hasattr(bpy.ops, 'import_scene') and hasattr(bpy.ops.import_scene, 'obj'):
                        bpy.ops.import_scene.obj(filepath=filepath)
                    else:
                        bpy.ops.import_scene.obj(filepath=filepath)
                except Exception as e:
                    print(f"OBJ 导入失败，尝试其他方法: {e}")
                    try:
                        bpy.ops.wm.obj_import(filepath=filepath)
                    except Exception as final_e:
                        print(f"最终OBJ导入失败: {final_e}")

        blender_compat = BlenderVersionCompat()
        
        # 创建渲染实例
        renderer = BlenderRenderClass()
        renderer.FLAGS = local_flags  # 传递flags到渲染实例
        
        # 设置全局blender_compat变量供renderer使用
        import __main__
        __main__.blender_compat = blender_compat
        
        # 执行单物体渲染
        success = renderer.render_single_object(cycle_id, scene_id, obj_index)
        
        return (True, cycle_id, scene_id, obj_index, "渲染成功")
        
    except Exception as e:
        error_msg = f"渲染失败: {str(e)}"
        print(f"⚠️ 循环{cycle_id}-场景{scene_id}-物体{obj_index}: {error_msg}")
        return (False, cycle_id, scene_id, obj_index, error_msg)

if FLAGS.input_type == 'continuous_scences':
    try:
        CYCLE_idx_list = parse_range_or_single(FLAGS.cycle_list)
        SCENE_idx_list = parse_range_or_single(FLAGS.scene_list)
    except ValueError as e:
        print(f"参数解析错误: {e}")
        sys.exit(1)
else:
    # 离散场景模式：从失败列表生成循环-场景配对
    CYCLE_idx_list, SCENE_idx_list = generate_cycle_scene_lists(failed_cycles_scenes)
    print(f"🎯 离散场景模式: 将重新渲染 {len(CYCLE_idx_list)} 个失败的循环-场景配对")

print("CYCLE_idx_list")
print(CYCLE_idx_list )
print("SCENE_idx_list")
print(SCENE_idx_list)

import logging

logger = logging.getLogger("bpy")
logger.setLevel(logging.WARNING)  # 设置Blender日志等级为WARNING或ERROR, 减少输出

import os
import sys
import argparse

# 获取数据集根目录
FILE_DIR = FLAGS.data_dir

# w10 可视化时候需要多加一句
# FILE_DIR = os.path.dirname(FILE_DIR)

import bpy
import csv
import json

import numpy as np
import shutil
from math import radians
import math
# import yaml
# from easydict import EasyDict
import csv

# Blender 版本兼容性辅助类
class BlenderVersionCompat:
    def __init__(self):
        self.version = bpy.app.version
        self.is_new_version = self.version >= (2, 80, 0)
        print(f"检测到 Blender 版本: {self.version[0]}.{self.version[1]}.{self.version[2]}")

    def select_object(self, obj, state=True):
        if self.is_new_version:
            obj.select_set(state)
        else:
            obj.select = state

    def get_light_type(self):
        return 'LIGHT' if self.is_new_version else 'LAMP'

    def get_light_add_op(self):
        return bpy.ops.object.light_add if self.is_new_version else bpy.ops.object.lamp_add

    def import_obj_file(self, filepath):
        try:
            if hasattr(bpy.ops.wm, 'obj_import'):
                bpy.ops.wm.obj_import(filepath=filepath)
                # print("使用新版本 OBJ 导入 API (3.0+)")
            elif hasattr(bpy.ops, 'import_scene') and hasattr(bpy.ops.import_scene, 'obj'):
                bpy.ops.import_scene.obj(filepath=filepath)
                # print("使用中版本 OBJ 导入 API (2.80-2.93)")
            else:
                bpy.ops.import_scene.obj(filepath=filepath)
                # print("使用旧版本 OBJ 导入 API (2.79-)")
        except Exception as e:
            print(f"OBJ 导入失败，尝试其他方法: {e}")
            try:
                bpy.ops.wm.obj_import(filepath=filepath)
            except Exception as final_e:
                print(f"最终OBJ导入失败: {final_e}")

# 创建全局兼容性对象
blender_compat = BlenderVersionCompat()

OBJ_PATH =  os.path.join(FILE_DIR, 'OBJ')
OUTDIR_physics_result_dir =  os.path.join(FILE_DIR, 'physics_result')
OUTDIR_dir_segment_images =  os.path.join(FILE_DIR, 'segment_images_single')

if not os.path.exists(OUTDIR_dir_segment_images):
    os.makedirs(OUTDIR_dir_segment_images)

class BlenderRenderClass:
    def __init__(self):
        # 将FLAGS设置为实例变量，以便在多进程中使用
        self.FLAGS = FLAGS if 'FLAGS' in globals() else None
        
        camera_info_path = self.FLAGS.camera_info_file if self.FLAGS else 'camera_info.yaml'
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
        
        # # 快速模式下降低分辨率
        # if FLAGS.fast_mode:
        #     self.img_w = int(self.img_w * 0.5)  # 降低50%分辨率
        #     self.img_h = int(self.img_h * 0.5)
        #     print(f"快速模式：分辨率降至 {self.img_w}x{self.img_h}")
        # elif FLAGS.ultra_fast:
        #     self.img_w = int(self.img_w * 0.25)  # 极速模式：降低75%分辨率
        #     self.img_h = int(self.img_h * 0.25)
        #     print(f"极速模式：分辨率降至 {self.img_w}x{self.img_h}")
            
        self.CAMERA_RESOLUTION = [int(self.img_w), int(self.img_h)]
        # self.DEPTH_DIVIDE = depth_graph_divide
        # self.DEPTH_LESS = depth_graph_less
        unit_of_obj = 'mm'
        if unit_of_obj == 'mm':
            self.meshScale = [0.001, 0.001, 0.001]  # 毫米转米
        elif unit_of_obj == 'm':
            self.meshScale = [1, 1, 1]

    def set_device(self):
        if self.FLAGS and self.FLAGS.use_gpu:
            try:
                bpy.context.scene.cycles.device = 'GPU'
                prefs = bpy.context.preferences.addons['cycles'].preferences
                prefs.compute_device_type = 'CUDA'
                prefs.get_devices()
                for device in prefs.devices:
                    if device.type == 'CUDA' or device.type == 'OPTIX':
                        device.use = True
                print('已启用NVIDIA GPU加速渲染')
            except Exception as e:
                print(f'GPU设置失败，回退到CPU渲染: {e}')
                bpy.context.scene.cycles.device = 'CPU'
        else:
            bpy.context.scene.cycles.device = 'CPU'
            print('已设置为CPU渲染')

    def camera_set(self):
        # 统一使用Cycles引擎渲染
        bpy.data.scenes["Scene"].render.engine = "CYCLES"
        bpy.data.scenes["Scene"].cycles.progressive = "BRANCHED_PATH"
        bpy.data.scenes["Scene"].cycles.aa_samples = 1
        bpy.data.scenes["Scene"].cycles.preview_aa_samples = 1
        print("使用Cycles引擎渲染")
            
        bpy.data.scenes["Scene"].render.resolution_x = self.CAMERA_RESOLUTION[0]
        bpy.data.scenes["Scene"].render.resolution_y = self.CAMERA_RESOLUTION[1]
        bpy.data.scenes["Scene"].render.resolution_percentage = 100
        
        # # 极速模式下进一步优化渲染设置
        # if FLAGS.ultra_fast:
        #     bpy.data.scenes["Scene"].render.resolution_percentage = 50  # 再次降低分辨率
        #     bpy.data.scenes["Scene"].render.pixel_aspect_x = 2.0  # 降低像素精度
        #     bpy.data.scenes["Scene"].render.pixel_aspect_y = 2.0
        #     print("极速模式：额外降低渲染精度")
            
        bpy.data.cameras["Camera"].type = "PERSP"
        bpy.data.cameras["Camera"].lens = self.CAMERA_FOCAL_LEN
        bpy.data.cameras["Camera"].lens_unit = "MILLIMETERS"
        bpy.data.cameras["Camera"].sensor_width = self.CAMERA_SENSOR_SIZE[0]
        bpy.data.cameras["Camera"].sensor_height = self.CAMERA_SENSOR_SIZE[1]
        bpy.data.cameras["Camera"].sensor_fit = "HORIZONTAL"
        
        # if not FLAGS.ultra_fast:
        #     bpy.data.scenes["Scene"].render.pixel_aspect_x = 1.0
        #     bpy.data.scenes["Scene"].render.pixel_aspect_y = self.CAMERA_SENSOR_SIZE[1] * self.CAMERA_RESOLUTION[0] / \
        #                                                      self.CAMERA_RESOLUTION[1] / self.CAMERA_SENSOR_SIZE[0]
       
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

    def read_csv(self, csv_path):      
        # 读取csv文件, 返回物体名称、位姿、索引
        with open(csv_path,'r') as csv_file:  
            all_lines=csv.reader(csv_file) 
            # 过滤空行
            list_file = [row for row in all_lines if any(row)]  
        array_file = np.array(list_file)[1:]
        obj_name = array_file[:,0]
        obj_index = array_file[:,1].astype('int')
        pose = array_file[:,2:9].astype('float32')
        return obj_name, pose, obj_index

    def import_obj(self, obj_name, pose, instance_index):
        # 导入指定物体到Blender场景中, 并设置其位姿
        
        # 获取路径信息
        FILE_DIR = self.FLAGS.data_dir if self.FLAGS else 'G:/Diffusion_Suction_DataSet'
        OBJ_PATH = os.path.join(FILE_DIR, 'OBJ')
        
        for o in bpy.data.objects:
            if o.type == 'MESH':
                blender_compat.select_object(o, True)
            else:
                blender_compat.select_object(o, False)
        bpy.ops.object.delete()  # 删除场景中所有网格对象

        for instance_index_ in instance_index:
            file_path = os.path.join(OBJ_PATH, obj_name[instance_index_] ,'object.obj')
            blender_compat.import_obj_file(file_path)
            instance = bpy.context.selected_objects[0]
            print(bpy.context.selected_objects)
            print(instance_index_)
            if not (self.FLAGS and self.FLAGS.disable_print):
                print(f"导入物体: {obj_name[instance_index_]}, 索引: {instance_index_}")
            instance.pass_index = instance_index_
            instance.scale = [0.001, 0.001, 0.001]  # 设置缩放(毫米转米)
            instance.location = [pose[instance_index_][0], pose[instance_index_][1], pose[instance_index_][2]]
            instance.rotation_mode = 'QUATERNION'
            instance.rotation_quaternion = [pose[instance_index_][3], pose[instance_index_][4], pose[instance_index_][5], pose[instance_index_][6]]
            
    # 使用节点配置深度图和分割图的输出
    def depth_graph(self, depth_path, segment_path):
        # 启用节点合成功能
        bpy.data.scenes["Scene"].use_nodes = 1

        # 定义合成节点
        scene = bpy.context.scene
        nodes = scene.node_tree.nodes
        links = scene.node_tree.links

        # 只在节点数量异常时清空，正常复用已配置节点
        if len(nodes) < 2 or not any(n.type == 'OUTPUT_FILE' for n in nodes):
            for node in nodes:
                nodes.remove(node)
            render_layers = nodes.new("CompositorNodeRLayers")
            output_file_label = nodes.new("CompositorNodeOutputFile")
            output_file_label.base_path = segment_path
            
            # 根据self.FLAGS.save_img_type设置输出格式
            if self.FLAGS and self.FLAGS.save_img_type.lower() == 'png':
                # PNG格式，适合快速保存mask
                output_file_label.format.file_format = "PNG"
                output_file_label.format.color_mode = "BW"  # 黑白模式，适合mask
                output_file_label.format.color_depth = '8'   # 8位深度
                output_file_label.format.compression = 15    # PNG压缩级别
                print("使用PNG格式保存mask，渲染速度更快")
            else:
                # 默认EXR格式
                output_file_label.format.file_format = "OPEN_EXR"
                output_file_label.format.color_mode = "RGB"
                output_file_label.format.color_depth = '32'
                print("使用EXR格式保存分割图像")
                
            links.new(render_layers.outputs['Image'], output_file_label.inputs['Image'])
        else:
            output_file_label = [n for n in nodes if n.type == 'OUTPUT_FILE'][0]
            output_file_label.base_path = segment_path
            
            # 更新现有节点的格式设置
            if self.FLAGS and self.FLAGS.save_img_type.lower() == 'png':
                output_file_label.format.file_format = "PNG"
                output_file_label.format.color_mode = "BW"
                output_file_label.format.color_depth = '8'
                output_file_label.format.compression = 15
            else:
                output_file_label.format.file_format = "OPEN_EXR"
                output_file_label.format.color_mode = "RGB"
                output_file_label.format.color_depth = '32'

    # 定义物体的材质(如颜色), 并让所有物体指向同一个材质
    def label_graph(self, label_number):
        # 遍历场景中的所有物体
        for obj in bpy.context.scene.objects:
            # 只处理网格对象
            if obj.type == 'MESH':
                # 清空物体的所有材质槽
                if obj.data.materials:
                    obj.data.materials.clear()
       
        mymat = bpy.data.materials.get('mymat')
        if not mymat:
            mymat = bpy.data.materials.new('mymat')
            mymat.use_nodes = True

        nodes = mymat.node_tree.nodes
        links = mymat.node_tree.links
        
        if self.FLAGS and self.FLAGS.save_img_type.lower() == 'png':
            # PNG mask使用简化材质，只需纯白色
            if len(nodes) < 2 or not any(n.type == 'EMISSION' for n in nodes):
                for node in nodes:
                    nodes.remove(node)
                OutputMat = nodes.new(type="ShaderNodeOutputMaterial")
                Emission = nodes.new(type="ShaderNodeEmission")
                # 直接设置为白色，简化计算
                Emission.inputs[0].default_value = (1.0, 1.0, 1.0, 1.0)  # 纯白色
                links.new(Emission.outputs[0], OutputMat.inputs[0])
        else:
            # EXR格式使用原来的复杂材质
            if len(nodes) < 2 or not any(n.type == 'EMISSION' for n in nodes):
                for node in nodes:
                    nodes.remove(node)
                # 配置颜色渐变节点
                ColorRamp = nodes.new(type="ShaderNodeValToRGB")
                ColorRamp.color_ramp.interpolation = 'LINEAR'
                ColorRamp.color_ramp.color_mode = 'RGB'
                ColorRamp.color_ramp.elements[0].color[:3] = [1.0, 0.0, 0.0]  # 红色
                ColorRamp.color_ramp.elements[1].color[:3] = [1.0, 1.0, 0.0]  # 黄色
                ObjectInfo = nodes.new(type="ShaderNodeObjectInfo")
                OutputMat = nodes.new(type="ShaderNodeOutputMaterial")
                Emission = nodes.new(type="ShaderNodeEmission")
                Math = nodes.new(type="ShaderNodeMath")
                Math.operation = "DIVIDE"
                Math.inputs[1].default_value = label_number
                # 连接ObjectInfo的Object Index输出（outputs[3]）到Math节点，实现分割标签的唯一性
                links.new(ObjectInfo.outputs[3], Math.inputs[0])  # Object Index（pass_index）/最大值
                links.new(Math.outputs[0], ColorRamp.inputs[0])
                links.new(ColorRamp.outputs[0], Emission.inputs[0])
                links.new(Emission.outputs[0], OutputMat.inputs[0])

        # 让所有网格对象都使用同一个材质
        objects = bpy.data.objects
        for obj in objects:
            if obj.type == 'MESH':
                if not 'mymat' in obj.data.materials:
                    obj.data.materials.append(mymat)

    def render_single_object(self, cycle_id, scene_id, obj_index):
        """
        渲染单个物体（用于多进程调用）
        
        Args:
            cycle_id: 循环ID
            scene_id: 场景ID
            obj_index: 物体索引
            
        Returns:
            bool: 渲染是否成功
        """
        try:
            # 设置设备和相机
            self.set_device()
            self.camera_set()
            
            # 获取文件路径
            FILE_DIR = self.FLAGS.data_dir if self.FLAGS else 'G:/Diffusion_Suction_DataSet'
            OUTDIR_physics_result_dir = os.path.join(FILE_DIR, 'physics_result')
            OUTDIR_dir_segment_images = os.path.join(FILE_DIR, 'segment_images_single')
            
            csv_path = os.path.join(OUTDIR_physics_result_dir, 'cycle_{:0>4}'.format(cycle_id),
                                  "{:0>3}".format(scene_id), "{:0>3}.csv".format(scene_id))
            obj_names, pose, segment_indexs = self.read_csv(csv_path)
            
            # 只渲染指定的物体
            obj_name = [obj_names[obj_index]]
            self.import_obj(obj_names, pose, [obj_index])
            
            segment_scene_path = os.path.join(OUTDIR_dir_segment_images, 'cycle_{:0>4}'.format(cycle_id),
                                            "{:0>3}".format(scene_id),
                                            "{:0>3}".format(scene_id)+"_{:0>3}".format(obj_index))
            depth_scene_path = segment_scene_path
            
            os.makedirs(depth_scene_path, exist_ok=True)
            os.makedirs(segment_scene_path, exist_ok=True)
            
            self.depth_graph(depth_scene_path, segment_scene_path)
            self.label_graph(len(obj_name) - 1)
            
            # 极速模式设置
            if self.FLAGS and self.FLAGS.ultra_fast:
                try:
                    bpy.context.scene.render.use_motion_blur = False
                    bpy.context.scene.render.use_border = False
                    bpy.context.scene.render.use_crop_to_border = False
                    bpy.context.scene.cycles.use_denoising = False if hasattr(bpy.context.scene.cycles, 'use_denoising') else None
                    bpy.context.scene.view_settings.view_transform = 'Standard'
                    bpy.context.scene.sequencer_colorspace_settings.name = 'sRGB'
                except Exception as e:
                    print(f"极速模式设置遇到错误，忽略: {e}")
            
            # 执行渲染
            bpy.ops.render.render()
            
            # 清理内存
            if self.FLAGS and (self.FLAGS.fast_mode or self.FLAGS.ultra_fast):
                bpy.ops.outliner.orphans_purge(do_recursive=True)
                if self.FLAGS.ultra_fast:
                    gc.collect()
            
            return True
            
        except Exception as e:
            print(f"单物体渲染失败: {e}")
            return False

    def render_scenes(self): 
        self.set_device()     
        self.camera_set()  # 设置相机参数  
        
        # 收集所有渲染任务
        render_tasks = []
        
        # 检查是否为离散模式，如果是则使用配对方式处理
        if self.FLAGS and self.FLAGS.input_type == 'discrete_scences':
            if len(CYCLE_idx_list) != len(SCENE_idx_list):
                raise ValueError(f"循环列表长度({len(CYCLE_idx_list)})与场景列表长度({len(SCENE_idx_list)})不匹配")
            
            # 离散模式：按配对处理
            for i, (cycle_id, scene_id) in enumerate(zip(CYCLE_idx_list, SCENE_idx_list)):
                print(f"🔄 准备第 {i+1}/{len(CYCLE_idx_list)} 个配对: 循环{cycle_id}-场景{scene_id}")
                tasks = self.prepare_cycle_scene_tasks(cycle_id, scene_id)
                render_tasks.extend(tasks)
        else:
            # 连续模式：原有的双重循环
            for cycle_id in CYCLE_idx_list:
                for scene_id in SCENE_idx_list:
                    tasks = self.prepare_cycle_scene_tasks(cycle_id, scene_id)
                    render_tasks.extend(tasks)

        # 决定是否使用多进程
        num_workers = self.FLAGS.num_workers if self.FLAGS else 1
        
        if num_workers > 1 and len(render_tasks) > 1:
            # 多进程渲染
            print(f"🚀 启动多进程渲染，使用 {num_workers} 个进程处理 {len(render_tasks)} 个任务")
            self.render_with_multiprocessing(render_tasks, num_workers)
        else:
            # 单进程渲染（原有方式）
            print(f"🎯 使用单进程渲染 {len(render_tasks)} 个任务")
            self.render_with_single_process(render_tasks)

        print('渲染完成!')
        
    def prepare_cycle_scene_tasks(self, cycle_id, scene_id):
        """
        准备单个循环-场景的所有渲染任务
        
        Returns:
            list: 渲染任务列表
        """
        tasks = []
        
        try:
            FILE_DIR = self.FLAGS.data_dir if self.FLAGS else 'G:/Diffusion_Suction_DataSet'
            OUTDIR_physics_result_dir = os.path.join(FILE_DIR, 'physics_result')
            
            csv_path = os.path.join(OUTDIR_physics_result_dir, 'cycle_{:0>4}'.format(cycle_id),
                                  "{:0>3}".format(scene_id), "{:0>3}.csv".format(scene_id))
            obj_names, pose, segment_indexs = self.read_csv(csv_path)
            
            # 为每个物体创建任务
            for obj_index in segment_indexs:
                # 将FLAGS转换为字典，以便在子进程中使用
                flags_dict = {}
                if self.FLAGS:
                    for attr in dir(self.FLAGS):
                        if not attr.startswith('_'):
                            flags_dict[attr] = getattr(self.FLAGS, attr)
                
                task = (cycle_id, scene_id, obj_index, flags_dict)
                tasks.append(task)
                
        except Exception as e:
            print(f"⚠️ 准备任务失败 循环{cycle_id}-场景{scene_id}: {e}")
            
        return tasks
        
    def render_with_multiprocessing(self, render_tasks, num_workers):
        """
        使用多进程进行渲染
        """
        print(f"📊 任务统计: 总共 {len(render_tasks)} 个物体需要渲染")
        
        start_time = time.time()
        successful_tasks = 0
        failed_tasks = 0
        
        try:
            with Pool(processes=num_workers) as pool:
                # 使用进度追踪
                results = []
                for i, task in enumerate(render_tasks):
                    result = pool.apply_async(render_single_object_task, (task,))
                    results.append(result)
                
                # 等待所有任务完成并显示进度
                for i, result in enumerate(results):
                    try:
                        success, cycle_id, scene_id, obj_index, message = result.get(timeout=300)  # 5分钟超时
                        if success:
                            successful_tasks += 1
                            print(f"✅ [{i+1}/{len(render_tasks)}] 循环{cycle_id}-场景{scene_id}-物体{obj_index}: {message}")
                        else:
                            failed_tasks += 1
                            print(f"❌ [{i+1}/{len(render_tasks)}] 循环{cycle_id}-场景{scene_id}-物体{obj_index}: {message}")
                    except Exception as e:
                        failed_tasks += 1
                        print(f"💥 [{i+1}/{len(render_tasks)}] 任务超时或异常: {e}")
                        
        except Exception as e:
            print(f"多进程渲染异常: {e}")
            
        end_time = time.time()
        total_time = end_time - start_time
        
        print(f"\n📈 渲染统计:")
        print(f"   ✅ 成功: {successful_tasks} 个")
        print(f"   ❌ 失败: {failed_tasks} 个")
        print(f"   ⏱️  总时间: {total_time:.2f} 秒")
        print(f"   ⚡ 平均速度: {len(render_tasks)/total_time:.2f} 个/秒")
        
    def render_with_single_process(self, render_tasks):
        """
        使用单进程进行渲染（保持原有逻辑）
        """
        start_time = time.time()
        successful_tasks = 0
        failed_tasks = 0
        
        for i, (cycle_id, scene_id, obj_index, flags_dict) in enumerate(render_tasks):
            try:
                print(f"🔄 [{i+1}/{len(render_tasks)}] 处理: 循环{cycle_id}-场景{scene_id}-物体{obj_index}")
                
                # 直接调用原有的单物体渲染逻辑
                success = self.render_single_object(cycle_id, scene_id, obj_index)
                
                if success:
                    successful_tasks += 1
                    print(f"✅ 循环{cycle_id}-场景{scene_id}-物体{obj_index}: 渲染成功")
                else:
                    failed_tasks += 1
                    print(f"❌ 循环{cycle_id}-场景{scene_id}-物体{obj_index}: 渲染失败")
                    
            except Exception as e:
                failed_tasks += 1
                print(f"💥 循环{cycle_id}-场景{scene_id}-物体{obj_index}: 异常 - {e}")
        
        end_time = time.time()
        total_time = end_time - start_time
        
        print(f"\n📈 渲染统计:")
        print(f"   ✅ 成功: {successful_tasks} 个")
        print(f"   ❌ 失败: {failed_tasks} 个")
        print(f"   ⏱️  总时间: {total_time:.2f} 秒")
        print(f"   ⚡ 平均速度: {len(render_tasks)/total_time:.2f} 个/秒")

    def process_single_cycle_scene(self, cycle_id, scene_id):
        """
        处理单个循环-场景组合的渲染（保留用于向后兼容）
        """
        print( 'cycle_id={} '.format(cycle_id)+'scene_id={}'.format(scene_id))
        
        FILE_DIR = self.FLAGS.data_dir if self.FLAGS else 'G:/Diffusion_Suction_DataSet'
        OUTDIR_physics_result_dir = os.path.join(FILE_DIR, 'physics_result')
        OUTDIR_dir_segment_images = os.path.join(FILE_DIR, 'segment_images_single')
        
        csv_path = os.path.join(OUTDIR_physics_result_dir, 'cycle_{:0>4}'.format(cycle_id),"{:0>3}".format(scene_id), "{:0>3}.csv".format(scene_id))
        obj_names, pose, segment_indexs = self.read_csv(csv_path)

        for i in segment_indexs:
            obj_name = []
            obj_name.append(obj_names[i])
            self.import_obj(obj_names, pose, [i])  # 只导入当前物体

            segment_scene_path = os.path.join(OUTDIR_dir_segment_images, 'cycle_{:0>4}'.format(cycle_id),"{:0>3}".format(scene_id),"{:0>3}".format(scene_id)+"_{:0>3}".format(i))
            depth_scene_path = segment_scene_path # 实际未用到
            if not os.path.exists(depth_scene_path):
                os.makedirs(depth_scene_path)
            if not os.path.exists(segment_scene_path):
                os.makedirs(segment_scene_path)
            
            self.depth_graph(depth_scene_path, segment_scene_path)  # 配置节点输出
            # 只渲染rgb图, 速度较快
            self.label_graph(len(obj_name) - 1)
            
            # 极速模式：禁用不必要的Blender功能
            if self.FLAGS and self.FLAGS.ultra_fast:
                try:
                    bpy.context.scene.render.use_motion_blur = False
                    bpy.context.scene.render.use_border = False
                    bpy.context.scene.render.use_crop_to_border = False
                    bpy.context.scene.cycles.use_denoising = False if hasattr(bpy.context.scene.cycles, 'use_denoising') else None
                    # 禁用所有后处理，使用最简单的色彩空间
                    bpy.context.scene.view_settings.view_transform = 'Standard'
                    bpy.context.scene.sequencer_colorspace_settings.name = 'sRGB'
                except Exception as e:
                    print(f"极速模式设置遇到错误，忽略: {e}")
                    
            # 使用try-catch保护渲染过程
            try:
                bpy.ops.render.render()  # 执行渲染
                print("渲染完成")
            except Exception as e:
                raise e
            
            # 每个物体渲染后立即清理内存，防止内存累积
            if self.FLAGS and (self.FLAGS.fast_mode or self.FLAGS.ultra_fast):
                bpy.ops.outliner.orphans_purge(do_recursive=True)
                # 极速模式：更频繁的内存清理
                if self.FLAGS.ultra_fast:
                    gc.collect()
        
        # 主动清理未使用的数据块和垃圾回收
        bpy.ops.outliner.orphans_purge(do_recursive=True)
        gc.collect()

if __name__ == '__main__':
    import time
    start_time = time.time()

    # 检查多进程兼容性
    if FLAGS.num_workers > 1:
        available_cores = cpu_count()
        if FLAGS.num_workers > available_cores:
            print(f"⚠️ 警告: 指定的进程数({FLAGS.num_workers})超过CPU核心数({available_cores})，将限制为{available_cores}")
            FLAGS.num_workers = available_cores
        
        print(f"🚀 多进程模式: 将使用 {FLAGS.num_workers} 个进程进行并行渲染")
        print(f"💡 提示: 每个进程将启动独立的Blender实例，请确保有足够的内存")
        
        # 多进程模式下的额外检查
        try:
            # 测试是否可以创建进程池
            from multiprocessing import Pool
            with Pool(1) as test_pool:
                pass
            print("✅ 多进程环境检查通过")
        except Exception as e:
            print(f"❌ 多进程环境检查失败: {e}")
            print("🔄 回退到单进程模式")
            FLAGS.num_workers = 1
    else:
        print("🎯 单进程模式: 使用传统的串行渲染")

    blender_generator = BlenderRenderClass()
    blender_generator.render_scenes()
    end_time = time.time()
    
    total_time = end_time - start_time
    print(f"\n🎉 全部渲染完成!")
    print(f"⏱️  总耗时: {total_time:.2f} 秒 ({total_time/60:.2f} 分钟)")
    
    if FLAGS.num_workers > 1:
        print(f"⚡ 使用了 {FLAGS.num_workers} 个并行进程")
    else:
        print("🔄 使用了单进程模式")