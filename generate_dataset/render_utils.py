# -*- coding:utf-8 -*-
"""
本文件用于在Blender中批量渲染三维场景, 自动导入物体模型、设置相机与光源参数, 并输出RGB图像、深度图和分割标签图。适用于数据集的自动生成与仿真渲染流程。
@author: Huang Dingtao
@checked: Huang Dingtao
"""

import sys
import os
import argparse
import re

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
# import yaml
# from easydict import EasyDict
import csv

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
                print("使用新版本 OBJ 导入 API (3.0+)")
            # 尝试中等版本的导入方式 (Blender 2.80-2.93)
            elif hasattr(bpy.ops, 'import_scene') and hasattr(bpy.ops.import_scene, 'obj'):
                bpy.ops.import_scene.obj(filepath=filepath)
                print("使用中版本 OBJ 导入 API (2.80-2.93)")
            # 旧版本的导入方式 (Blender 2.79-)
            else:
                bpy.ops.import_scene.obj(filepath=filepath)
                print("使用旧版本 OBJ 导入 API (2.79-)")
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
    def __init__(self, ):
        # *****Blender 相机参数设置*****
        resolution = [1920, 1200]  # 渲染分辨率
        focal_length = 16.0        # 相机焦距, 单位mm
        sensor_size =  [11.25, 7.03]  # 相机传感器尺寸, 单位mm
        cam_location_x_y_z = [0, 0, 1.70]  # 相机在三维空间的位置
        # cam_rotation_qw_qx_qy_qz = [0, -0.70710678, -0.70710678, 0] # [90. ,  0. ,180.]
        cam_rotation_qw_qx_qy_qz = [0.000000e+00 ,0.000000e+00 ,1.000000e+00 ,6.123234e-17]  # [0. ,  0. ,180.]
        
        depth_graph_divide =  2    # 深度图缩放因子
        depth_graph_less = 3       # 深度图阈值

        self.CAMERA_RESOLUTION = resolution
        self.CAMERA_FOCAL_LEN = focal_length
        self.CAMERA_SENSOR_SIZE = sensor_size
        self.CAMERA_LOCATION = cam_location_x_y_z
        self.CAMERA_ROTATION = cam_rotation_qw_qx_qy_qz
        self.DEPTH_DIVIDE = depth_graph_divide
        self.DEPTH_LESS = depth_graph_less
        # *****Blender 相机参数设置*****
        unit_of_obj='mm'
        if unit_of_obj == 'mm':
            self.meshScale = [0.001, 0.001, 0.001]  # 毫米转米
        elif unit_of_obj == 'm':
            self.meshScale = [1, 1, 1]

    def camera_set(self):
        # 自动启用NVIDIA GPU加速
        bpy.context.scene.cycles.device = 'GPU'
        prefs = bpy.context.preferences.addons['cycles'].preferences
        prefs.compute_device_type = 'CUDA'  # 如果支持OPTIX可改为'OPTIX'
        prefs.get_devices()
        for device in prefs.devices:
            if device.type == 'CUDA' or device.type == 'OPTIX':
                device.use = True
        print('已启用NVIDIA GPU加速渲染')

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
       
        bpy.data.objects["Camera"].location = [self.CAMERA_LOCATION[0],
                                               self.CAMERA_LOCATION[1],
                                               self.CAMERA_LOCATION[2]]
        bpy.data.objects["Camera"].rotation_mode = 'QUATERNION'
        bpy.data.objects["Camera"].rotation_quaternion = [self.CAMERA_ROTATION[0],
                                                          self.CAMERA_ROTATION[1],
                                                          self.CAMERA_ROTATION[2],
                                                          self.CAMERA_ROTATION[3]]
        # 让相机坐标系绕X轴旋转180度, 适配Blender坐标系
        bpy.data.objects["Camera"].rotation_mode = 'XYZ'
        bpy.data.objects["Camera"].rotation_euler[0] = bpy.data.objects["Camera"].rotation_euler[0] + math.pi

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
            
            instance = bpy.context.selected_objects[0]
            print(bpy.context.selected_objects)
            print(instance_index_)
            instance.pass_index = instance_index_
            instance.scale = [0.001, 0.001, 0.001]  # 设置缩放(毫米转米)
            instance.location = [pose[instance_index_][0], pose[instance_index_][1], pose[instance_index_][2]]
            instance.rotation_mode = 'QUATERNION'
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
        ObjectInfo = nodes.new(type="ShaderNodeObjectInfo")
        OutputMat = nodes.new(type="ShaderNodeOutputMaterial")
        Emission = nodes.new(type="ShaderNodeEmission")

        Math = nodes.new(type="ShaderNodeMath")
        Math.operation = "DIVIDE"
        Math.inputs[1].default_value = label_number

        links.new(ObjectInfo.outputs[1], Math.inputs[0])
        links.new(Math.outputs[0], ColorRamp.inputs[0])
        links.new(ColorRamp.outputs[0], Emission.inputs[0])
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
        for cycle_id in CYCLE_idx_list:
            for scene_id in SCENE_idx_list:
                start_time = time.time()  # 记录起始时间戳
                self.camera_set()  # 设置相机和光源
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
                
        np.save('times.npy', times)
        print(f"总计渲染 {len(times)} 个场景，平均耗时: {np.mean(times):.2f}秒")
        print(times)
if __name__ == '__main__':
    import time
    
    blender_generator = BlenderRenderClass()
    blender_generator.render_scenes()

