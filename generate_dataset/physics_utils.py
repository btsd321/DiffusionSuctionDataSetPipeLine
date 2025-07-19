# -*- coding:utf-8 -*-
"""
本脚本用于批量生成物理仿真场景, 随机投放多个物体到箱体中, 并保存每个物体的最终位姿到csv文件。适用于数据集物理仿真数据的自动生成流程。

@author: Huang Dingtao
@checked: Huang Dingtao

"""
import os
import sys
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
# 循环次数
parser.add_argument('--cycle_list', type=str, default='100', help='循环编号，支持格式: "5"(单个), "[1,10]"(区间), "{1,3,5}"(列表)')
# 场景数量
parser.add_argument('--scene_list', type=str, default='50', help='场景编号，支持格式: "5"(单个), "[1,10]"(区间), "{1,3,5}"(列表)')
# 是否显示GUI界面
parser.add_argument('--visualize', action='store_true', help='设置该参数则显示GUI界面')
parser.add_argument('--input_type', type=str, default='continuous_scences', choices=['discrete_scences', 'continuous_scences'],help='输入类型，\
                    discrete_scences表示输入为离散场景此时忽略cycle_list参数和scene_list参数，continuous_scences表示连续场景此时cycle_list参数和scene_list参数为循环次数和场景数量')
FLAGS = parser.parse_args()

# 获取数据集根目录
FILE_DIR = FLAGS.data_dir

failed_cycles_scenes = [
    [1, 34],
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

if FLAGS.input_type == 'continuous_scences':
    CYCLE_idx_list = parse_range_or_single(FLAGS.cycle_list)
    SCENE_idx_list = parse_range_or_single(FLAGS.scene_list)   # 每个循环50个场景
else:
    # 离散场景模式：从失败列表生成循环-场景配对
    CYCLE_idx_list, SCENE_idx_list = generate_cycle_scene_lists(failed_cycles_scenes)
    print(f"🎯 离散场景模式: 将重新生成 {len(CYCLE_idx_list)} 个失败的循环-场景配对")

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
        
        # 检查是否为离散模式，如果是则使用配对方式处理
        if FLAGS.input_type == 'discrete_scences':
            if len(CYCLE_idx_list) != len(SCENE_idx_list):
                raise ValueError(f"循环列表长度({len(CYCLE_idx_list)})与场景列表长度({len(SCENE_idx_list)})不匹配")
            
            # 离散模式：按配对处理
            for i, (cycle_id, scene_id) in enumerate(zip(CYCLE_idx_list, SCENE_idx_list)):
                print(f"🔄 处理第 {i+1}/{len(CYCLE_idx_list)} 个配对: 循环{cycle_id}-场景{scene_id}")
                self.process_single_cycle_scene(cycle_id, scene_id)
        else:
            # 连续模式：原有的双重循环
            for cycle_id in CYCLE_idx_list:
                for scene_id in SCENE_idx_list:
                    self.process_single_cycle_scene(cycle_id, scene_id)

        print('The Simulation is finished!')
        
    def process_single_cycle_scene(self, cycle_id, scene_id):
        """
        处理单个循环-场景组合的物理仿真
        """
        self.scene_init()

        # 加载箱体的四个侧壁
        cube_ind_1 = pybullet.loadURDF(os.path.join(os.path.join(FILE_DIR, 'BOX'), 'cube1.urdf'), (0, self.box_length*0.5+self.box_thickness*0.5, self.box_height/2), pybullet.getQuaternionFromEuler([0, 0, 0]), useFixedBase=1)
        cube_ind_2 = pybullet.loadURDF(os.path.join(os.path.join(FILE_DIR, 'BOX'), 'cube1.urdf'), (0, -self.box_length*0.5-self.box_thickness*0.5, self.box_height/2), pybullet.getQuaternionFromEuler([0, 0, 0]), useFixedBase=1)
        cube_ind_3 = pybullet.loadURDF(os.path.join(os.path.join(FILE_DIR, 'BOX'), 'cube2.urdf'), (self.box_width*0.5+self.box_thickness*0.5, 0, self.box_height/2), pybullet.getQuaternionFromEuler([0, 0, 0]), useFixedBase=1)
        cube_ind_4 = pybullet.loadURDF(os.path.join(os.path.join(FILE_DIR, 'BOX'), 'cube2.urdf'), (-self.box_width*0.5-self.box_thickness*0.5, -0, self.box_height/2), pybullet.getQuaternionFromEuler([0, 0, 0]), useFixedBase=1)
        
        # 加载物体, 检查是否有物体超出箱体, 如果有则重新投放
        while (1):
            flag = 0
            # 随机投放物体到箱体中
            multi_body_objects_first_layer, name_list = self.random_drop_objects_single(self.meshScale, scene_id)
            # 检查物体是否超出箱体
            for sparepart_id in multi_body_objects_first_layer:
                final_position, angle = pybullet.getBasePositionAndOrientation(sparepart_id)
                # 检查物体z坐标是否超出箱体高度或低于底部
                if((math.fabs(final_position[2]) >= self.box_height) or (final_position[2] < 0)):    
                    flag = 1
                # 检查物体x坐标是否超出箱体宽度范围
                if(math.fabs(final_position[0]) >= (self.box_width / 2 + 0.3)):
                    flag = 1
                # 检查物体y坐标是否超出箱体长度范围
                if(math.fabs(final_position[1]) >= (self.box_length / 2 + 0.3)):
                    flag = 1
            if flag == 1:
                for i in multi_body_objects_first_layer:
                    pybullet.removeBody(i)
            if flag == 0:
                break
        # 计算当前场景中实际投放的物体数量(即物理仿真后箱体内的物体个数)。
        foreground_nums = len(multi_body_objects_first_layer)
        # 生成一个索引列表，内容是 [0, 1, ..., foreground_nums-1]，用于标记每个物体的编号。
        index_list = [i for i in range(foreground_nums)] 
        # 断言实际投放的物体数量必须等于当前场景编号 scene_id(即本场景要求的物体数量)。如果不相等，程序会报错，说明仿真结果与预期不符。
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
        # 自动收敛仿真：如果物体位姿变化很小则提前停止
        max_steps = 720
        min_steps = 100  # 最少仿真步数，防止太快结束
        stable_threshold = 1e-4  # 位置变化阈值（米）
        stable_rot_threshold = 1e-4  # 四元数变化阈值
        stable_count_required = 10  # 连续多少步都稳定才提前终止
        stable_count = 0
        prev_poses = [pybullet.getBasePositionAndOrientation(mb) for mb in multi_body_list]
        for step in range(max_steps):
            pybullet.stepSimulation()
            curr_poses = [pybullet.getBasePositionAndOrientation(mb) for mb in multi_body_list]
            # 判断所有物体是否都稳定
            all_stable = True
            for (prev, curr) in zip(prev_poses, curr_poses):
                pos_prev, quat_prev = prev
                pos_curr, quat_curr = curr
                pos_delta = np.linalg.norm(np.array(pos_prev) - np.array(pos_curr))
                quat_delta = np.linalg.norm(np.array(quat_prev) - np.array(quat_curr))
                if pos_delta > stable_threshold or quat_delta > stable_rot_threshold:
                    all_stable = False
                    break
            if all_stable and step > min_steps:
                stable_count += 1
            else:
                stable_count = 0
            prev_poses = curr_poses
            if stable_count >= stable_count_required:
                print(f"提前收敛，仿真步数: {step+101}, 节省了: {max_steps - step - 1} 步")
                break
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