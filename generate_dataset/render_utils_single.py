# -*- coding:utf-8 -*-
"""
本文件用于在Blender中对单个物体进行批量渲染, 自动导入物体模型、设置相机参数, 并输出分割标签图。适用于数据集单物体分割标签的自动生成与渲染流程。

@author: Huang Dingtao
@checked: Huang Dingtao

"""
import sys
import re

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
parser.add_argument('--use_gpu', action='store_true', help='设置该参数则启用GPU加速渲染')
FLAGS = parser.parse_args()

try:
    CYCLE_idx_list = parse_range_or_single(FLAGS.cycle_list)
    SCENE_idx_list = parse_range_or_single(FLAGS.scene_list)
except ValueError as e:
    print(f"参数解析错误: {e}")
    sys.exit(1)
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

OBJ_PATH =  os.path.join(FILE_DIR, 'OBJ')
OUTDIR_physics_result_dir =  os.path.join(FILE_DIR, 'physics_result')
OUTDIR_dir_segment_images =  os.path.join(FILE_DIR, 'segment_images_sigle')

if not os.path.exists(OUTDIR_dir_segment_images):
    os.makedirs(OUTDIR_dir_segment_images)

class BlenderRenderClass:
    def __init__(self, ):
        # *****Blender 相机参数设置*****
        resolution = [1920, 1200]  # 渲染分辨率
        focal_length = 16.0        # 相机焦距, 单位mm
        sensor_size =  [11.25, 7.03]  # 相机传感器尺寸, 单位mm
        cam_location_x_y_z = [0, 0, 1.70]  # 相机在三维空间的位置
        cam_rotation_qw_qx_qy_qz =  [0.000000e+00 ,0.000000e+00 ,1.000000e+00 ,6.123234e-17]  # 相机四元数旋转
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
        if FLAGS.use_gpu:
            bpy.context.scene.cycles.device = 'GPU'
            prefs = bpy.context.preferences.addons['cycles'].preferences
            prefs.compute_device_type = 'CUDA'  # 如果支持OPTIX可改为'OPTIX'
            prefs.get_devices()
            for device in prefs.devices:
                if device.type == 'CUDA' or device.type == 'OPTIX':
                    device.use = True
            print('已启用NVIDIA GPU加速渲染')
        else:
            bpy.context.scene.cycles.device = 'CPU'
            print('已设置为CPU渲染')

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
        for o in bpy.data.objects:
            if o.type == 'MESH':
                o.select = True
            else:
                o.select = False
        bpy.ops.object.delete()  # 删除场景中所有网格对象

        for instance_index_ in instance_index:
            file_path = os.path.join(OBJ_PATH, obj_name[instance_index_] ,'object.obj')
            bpy.ops.import_scene.obj(filepath=file_path)
            instance = bpy.context.selected_objects[0]
            print(bpy.context.selected_objects)
            print(instance_index_)
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
        for node in nodes:
            nodes.remove(node)

        render_layers = nodes.new("CompositorNodeRLayers")

        # 输出分割图(OPEN_EXR格式, 32位RGB)
        output_file_label = nodes.new("CompositorNodeOutputFile")
        output_file_label.base_path = segment_path
        output_file_label.format.file_format = "OPEN_EXR"
        output_file_label.format.color_mode = "RGB"
        output_file_label.format.color_depth = '32'

        # 连接渲染输出到分割图输出节点
        links.new(render_layers.outputs['Image'], output_file_label.inputs['Image'])

    # 定义物体的材质(如颜色), 并让所有物体指向同一个材质
    def label_graph(self, label_number):
        # 遍历场景中的所有物体
        for obj in bpy.context.scene.objects:
            # 只处理网格对象
            if obj.type == 'MESH':
                # 清空物体的所有材质槽
                if obj.data.materials:
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
        for cycle_id in CYCLE_idx_list:

            for scene_id in SCENE_idx_list:
                print( 'cycle_id={}'.format(cycle_id)+'scene_id={}'.format(scene_id))
                self.camera_set()  # 设置相机参数
                
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
                    bpy.ops.render.render()  # 执行渲染

if __name__ == '__main__':
    import time
    start_time = time.time()

    blender_generator = BlenderRenderClass()
    blender_generator.render_scenes()
    end_time = time.time()
    print(end_time-start_time )