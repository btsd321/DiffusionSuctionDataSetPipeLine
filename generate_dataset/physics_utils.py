

# -*- coding:utf-8 -*-
"""
本脚本用于批量生成物理仿真场景, 随机投放多个物体到箱体中, 并保存每个物体的最终位姿到csv文件。适用于数据集物理仿真数据的自动生成流程。

@author: Huang Dingtao
@checked: Huang Dingtao

"""
import os
import sys
import argparse

# 命令行参数解析
parser = argparse.ArgumentParser()
# 数据集根目录
parser.add_argument('--data_dir', type=str, default='G:/Diffusion_Suction_DataSet', help='数据集根目录')
# 循环次数
parser.add_argument('--cycle_num', type=int, default=100, help='循环次数')
# 场景数量
parser.add_argument('--scene_num', type=int, default=50, help='场景数量')
# 是否显示GUI界面
parser.add_argument('--visualize', action='store_true', help='设置该参数则显示GUI界面')
FLAGS = parser.parse_args()

# 获取数据集根目录
FILE_DIR = FLAGS.data_dir
# 获取循环次数
CYCLE_NUM = FLAGS.cycle_num
# 获取场景数量
SCENE_NUM = FLAGS.scene_num

CYCLE_idx_list = range(0, CYCLE_NUM)  # 100个循环
SCENE_idx_list = range(1, SCENE_NUM + 1)   # 每个循环50个场景

# OBJ文件夹路径及物体名称列表(部分物体被排除)
OBJ_folder_path = os.path.join(FILE_DIR, "OBJ")
OBJ_files_and_dirs = os.listdir(OBJ_folder_path)
OBJ_name = [str(i) for i in range(0, 113)]
OBJ_name.remove("112")
OBJ_name.remove("110")
OBJ_name.remove("90")
OBJ_name.remove("91")
OBJ_name.remove("80")
OBJ_name.remove("81")
OBJ_name.remove("70")
OBJ_name.remove("10")
OBJ_name.remove("11")
OBJ_name.remove("20")
OBJ_name.remove("21")
OBJ_name.remove("30")
OBJ_name.remove("41")

print(OBJ_name)
print(len(OBJ_name))

OBJ_PATH = os.path.join(FILE_DIR, 'OBJ')
OUTDIR_dir = os.path.join(FILE_DIR, 'physics_result')
if not os.path.exists(OUTDIR_dir):
    os.makedirs(OUTDIR_dir)

# from tkinter.tix import ButtonBox
import pybullet
import time
import math
import pybullet_data
import csv
import numpy as np
import yaml
import random

class GenerateSimulationResult:
    def __init__(self):
        # meshScale用于控制物体模型的单位, 默认为毫米转米
        unit_of_obj = 'mm'
        if unit_of_obj == 'mm':
            self.meshScale = [0.001, 0.001, 0.001]  # 毫米转米
        elif unit_of_obj == 'm':
            self.meshScale = [1, 1, 1]

        # 设置箱体尺寸：宽800mm, 长600mm, 高500mm, 厚度50mm
        self.box_width  = 0.8
        self.box_length = 0.6
        self.box_thickness = 0.05
        self.box_height =  0.50
        
        # 是否显示GUI界面, 根据命令行参数决定
        self.show_GUI = 1 if FLAGS.visualize else 0
        # 随机投放物体的位置范围[x_min, y_min, z_min, z_max, box底部厚度]
        self.random_range = [0.1, 0.1, 0.13, 0.15]
        self.box_bottom_thickness = 0.01
        self.random_range.append(self.box_bottom_thickness)

    def scene_init(self):
        # 初始化仿真场景, 包括物理引擎、地面、重力等
        if self.show_GUI:
            _ = pybullet.connect(pybullet.GUI) 
        else:
            _ = pybullet.connect(pybullet.DIRECT)
        pybullet.setPhysicsEngineParameter(numSolverIterations=10) 
        pybullet.setTimeStep(1. / 120.)
        pybullet.setAdditionalSearchPath(pybullet_data.getDataPath())
        _ = pybullet.startStateLogging(pybullet.STATE_LOGGING_PROFILE_TIMINGS, "visualShapeBench.json")
        pybullet.loadURDF("plane100.urdf", useMaximalCoordinates=True)
        pybullet.configureDebugVisualizer(pybullet.COV_ENABLE_RENDERING, 0)
        pybullet.configureDebugVisualizer(pybullet.COV_ENABLE_GUI, 0)
        pybullet.configureDebugVisualizer(pybullet.COV_ENABLE_TINY_RENDERER, 0)
        pybullet.setGravity(0, 0, -5)
        # 设置相机参数
        pybullet.resetDebugVisualizerCamera(cameraDistance=0.5, cameraYaw=0, cameraPitch=-89.99, cameraTargetPosition=[0.01, 0.01, 0.01])

    def random_drop_objects_single(self, mesh_scale, nums):
        # 随机投放指定数量的物体到场景中
        multi_body = []
        OBJ_name_sample_ids = []
        
        for _ in range(nums):
            vShapedId = []
            cShapedId = []
            # 随机选择一个物体
            OBJ_name_sample_id = random.choice(range(len(OBJ_name)))
            OBJ_name_sample_ids.append(OBJ_name[OBJ_name_sample_id])
            file_path = os.path.join(OBJ_PATH, OBJ_name[OBJ_name_sample_id], 'object.obj')
            # 创建物体的可视形状和碰撞形状
            vShapedId = pybullet.createVisualShape(shapeType=pybullet.GEOM_MESH, fileName=file_path, meshScale=mesh_scale)
            cShapedId = pybullet.createCollisionShape(shapeType=pybullet.GEOM_MESH, fileName=file_path, meshScale=mesh_scale)
        
            position = []
            # 随机生成物体的投放位置
            position.append(np.random.uniform(-self.random_range[0], self.random_range[0]))
            position.append(np.random.uniform(-self.random_range[1], self.random_range[1]))
            position.append(np.random.uniform(self.random_range[2] + self.box_bottom_thickness, self.random_range[3] + self.box_bottom_thickness))
            # 随机生成欧拉角并转为四元数
            rand_euler_angle = np.random.uniform(-2.0 * math.pi, 2.0 * math.pi, [3])
            rand_quat = pybullet.getQuaternionFromEuler(rand_euler_angle)

            # 创建物体并添加到仿真中
            multi_body.append(pybullet.createMultiBody(
                baseMass=1,
                baseCollisionShapeIndex=cShapedId,
                baseVisualShapeIndex=vShapedId,
                basePosition=position,
                baseOrientation=rand_quat,
                useMaximalCoordinates=False))
            pybullet.changeVisualShape(multi_body[-1], -1, rgbaColor=[1, 0, 0, 1])
            pybullet.configureDebugVisualizer(pybullet.COV_ENABLE_RENDERING, 1)
        # 进行若干步仿真, 使物体稳定
        for _ in range(160):
            pybullet.stepSimulation()
            time.sleep(1. / 240)
        if self.show_GUI:
            for _ in range(1000):
                pybullet.stepSimulation()
                time.sleep(1. / 240)
            pybullet.configureDebugVisualizer(pybullet.COV_ENABLE_RENDERING, 0)
            
        return multi_body, OBJ_name_sample_ids

    def generate_single_object(self):    
        # 生成单场景多物体的物理仿真结果
        cycle_idx_list = CYCLE_idx_list
        scene_idx_list = SCENE_idx_list
        
        for cycle_id in cycle_idx_list:
            for scene_id in scene_idx_list:
                self.scene_init()

                # 加载箱体的四个侧壁
                cube_ind_1 = pybullet.loadURDF(os.path.join(os.path.join(FILE_DIR, 'BOX'), 'cube1.urdf'), (0, self.box_length*0.5+self.box_thickness*0.5, self.box_height/2), pybullet.getQuaternionFromEuler([0, 0, 0]), useFixedBase=1)
                cube_ind_2 = pybullet.loadURDF(os.path.join(os.path.join(FILE_DIR, 'BOX'), 'cube1.urdf'), (0, -self.box_length*0.5-self.box_thickness*0.5, self.box_height/2), pybullet.getQuaternionFromEuler([0, 0, 0]), useFixedBase=1)
                cube_ind_3 = pybullet.loadURDF(os.path.join(os.path.join(FILE_DIR, 'BOX'), 'cube2.urdf'), (self.box_width*0.5+self.box_thickness*0.5, 0, self.box_height/2), pybullet.getQuaternionFromEuler([0, 0, 0]), useFixedBase=1)
                cube_ind_4 = pybullet.loadURDF(os.path.join(os.path.join(FILE_DIR, 'BOX'), 'cube2.urdf'), (-self.box_width*0.5-self.box_thickness*0.5, -0, self.box_height/2), pybullet.getQuaternionFromEuler([0, 0, 0]), useFixedBase=1)
                
                # 加载物体, 检查是否有物体超出箱体, 如果有则重新投放
                while (1):
                    flag = 0
                    multi_body_objects_first_layer, name_list = self.random_drop_objects_single(self.meshScale, scene_id)
                    for sparepart_id in multi_body_objects_first_layer:
                        final_position, angle = pybullet.getBasePositionAndOrientation(sparepart_id)
                        # 检查物体z坐标是否超出箱体高度或低于底部
                        if((math.fabs(final_position[2]) >= self.box_height) or (final_position[2] < 0)):    
                            flag = 1
                    if flag == 1:
                        for i in multi_body_objects_first_layer:
                            pybullet.removeBody(i)
                    if flag == 0:
                        break
                foreground_nums = len(multi_body_objects_first_layer)
                index_list = [i for i in range(foreground_nums)] 
                assert foreground_nums == scene_id
                
                if self.show_GUI:
                    for _ in range(100):
                        pybullet.stepSimulation()
                        time.sleep(1. / 240)
                    pybullet.configureDebugVisualizer(pybullet.COV_ENABLE_RENDERING, 0)

                # 保存仿真结果
                self.save_results(cycle_id, scene_id, multi_body_objects_first_layer, index_list, name_list)
                pybullet.disconnect()

        print('The Simulation is finished!')      

    def save_results(self, cycle_id, scene_id, multi_body_list, index_list, name_list):
        # 保存当前循环和场景的仿真结果到csv文件
        headers = ["Type", "Index", "x", "y", "z", "w", "i", "j", "k"]
        rows = []
        # 让仿真再运行720步, 确保物体完全静止
        for _ in range(720):
            pybullet.stepSimulation()
        for i, mb in enumerate(multi_body_list):
            final_position, quat = pybullet.getBasePositionAndOrientation(mb)
            # 保存物体名称、索引、位置(x,y,z)、四元数(w,i,j,k)
            row = [name_list[i], index_list[i],
                   final_position[0], final_position[1],
                   final_position[2],
                   quat[3], quat[0], quat[1], quat[2]]
            rows.append(row)
        
        out_cycle_dir = os.path.join(OUTDIR_dir, 'cycle_{:0>4}'.format(cycle_id), "{:0>3}".format(scene_id))
        if not os.path.exists(out_cycle_dir):
            os.makedirs(out_cycle_dir)
        file_loc = out_cycle_dir + '/' + "{:0>3}.csv".format(scene_id)
        with open(file_loc, 'w', newline='') as f:
            f_csv = csv.writer(f)
            f_csv.writerow(headers)
            f_csv.writerows(rows)
        print(f' {cycle_id} cycle: {scene_id} scene is completed')
       
if __name__ == '__main__':
    import time
    start_time = time.time()  

    physics_generator = GenerateSimulationResult()
    physics_generator.generate_single_object()

    end_time = time.time()
    print(end_time-start_time )