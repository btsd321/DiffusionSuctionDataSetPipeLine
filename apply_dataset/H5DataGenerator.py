"""
本文件实现了机器人吸取任务的H5数据集生成器，是整个数据处理管道的核心组件。

主要功能：
1. 多模态数据融合：整合深度图像、分割图像、物体位姿真值、尺寸标签等数据
2. 点云生成与处理：从深度图像重建3D点云，进行采样和标准化
3. 法向量估计：使用Open3D计算点云表面法向量，并统一朝向
4. 多维度评分计算：
   - 密封评分(seal scores)：基于预训练模型和KNN算法
   - 抗扭评分(wrench scores)：考虑重力方向和吸盘姿态的物理建模
   - 可行性评分(feasibility scores)：基于几何碰撞检测
   - 尺寸评分：物体可见面积比例

核心算法：
- 深度图到点云的投影变换（相机内参标定）
- 最远点采样(FPS)进行点云下采样
- KDTree半径搜索进行法向量估计
- KNN最近邻查找进行密封评分插值
- 物理建模计算重力影响下的抗扭矩评分
- 几何碰撞检测评估吸取可行性

数据流程：
输入 → 点云重建 → 法向量估计 → 多维评分计算 → 可视化验证 → H5格式输出

应用场景：
- 机器人吸取任务的深度学习训练数据生成
- 多模态感知数据的标准化处理
- 物理约束下的抓取质量评估

@author: 数据处理管道核心模块
@version: 1.0
"""

import os
import sys
import threading
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'config'))
from camera_info import CameraInfo

# 导入必要的库
import json          # JSON配置文件解析
import math          # 数学运算
import numpy as np   # 数值计算
import cv2           # 图像处理
import os            # 操作系统接口
import time          # 时间测量
import torch         # 深度学习框架
import open3d as o3d # 3D点云处理
from torch_cluster import knn  # KNN最近邻搜索

# PointNet2操作库，用于点云采样
from pointnet2_ops_lib.pointnet2_ops.pointnet2_utils import furthest_point_sample
import open3d as o3d  # 3D几何处理
import h5py          # HDF5文件格式处理
import time          # 性能测量
import csv           # CSV文件处理
import matplotlib
from matplotlib.font_manager import FontProperties
import matplotlib.pyplot as plt

# 可视化设置
matplotlib.rcParams['axes.unicode_minus'] = False    # 负号正常显示

# 设置中文字体，处理字体文件不存在的情况
def setup_chinese_font():
    """
    设置中文字体，提供多种字体路径的回退机制
    """
    # 尝试多种可能的中文字体路径
    font_paths = [
        '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc',     # 新安装的Noto CJK字体
        '/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf',  # 系统中实际存在的中文字体
        '/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc',
        '/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc',
        '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',
        '/System/Library/Fonts/PingFang.ttc',  # macOS
        'C:\\Windows\\Fonts\\msyh.ttc',        # Windows
        'C:\\Windows\\Fonts\\simhei.ttf',      # Windows
    ]
    
    # 尝试每个字体路径
    for font_path in font_paths:
        if os.path.exists(font_path):
            try:
                font = FontProperties(fname=font_path)
                print(f"成功加载字体: {font_path}")
                return font
            except Exception as e:
                print(f"字体加载失败: {font_path}, 错误: {e}")
                continue
    
    # 最后的回退：使用默认字体
    print("警告: 未找到合适的中文字体，使用默认字体")
    return FontProperties()

# 设置matplotlib中文字体支持
def setup_matplotlib_chinese():
    """
    设置matplotlib的中文字体支持
    """
    # 使用简单的字体设置，避免复杂的字体族配置
    matplotlib.rcParams['axes.unicode_minus'] = False  # 负号正常显示
    
    # 清除matplotlib的字体缓存
    try:
        import matplotlib.font_manager as fm
        fm._rebuild()
    except:
        pass

# 初始化字体设置
setup_matplotlib_chinese()
font = setup_chinese_font()

def viewpoint_to_matrix_x(towards):
    """
    根据朝向向量生成以x轴为主的旋转矩阵
    
    用于将吸盘的朝向向量转换为3x3旋转矩阵，其中x轴为主方向。
    主要用于碰撞检测中的坐标系变换。
    
    参数:
        towards (numpy.ndarray): 朝向向量，形状为(3,)
    
    返回:
        numpy.ndarray: 3x3旋转矩阵
    """
    # 设置x轴为朝向方向
    axis_x = towards
    # 构造y轴，垂直于x轴在水平面内
    axis_y = np.array([-axis_x[1], axis_x[0], 0])
    # 标准化x轴和y轴
    axis_x = axis_x / np.linalg.norm(axis_x)
    axis_y = axis_y / np.linalg.norm(axis_y)
    # 通过叉积计算z轴
    axis_z = np.cross(axis_x, axis_y)
    # 组合成旋转矩阵
    R2 = np.c_[axis_x, np.c_[axis_y, axis_z]]
    matrix = R2
    return matrix

def viewpoint_to_matrix_z(towards):
    """
    根据朝向向量生成以z轴为主的旋转矩阵
    
    用于将法向量转换为3x3旋转矩阵，其中z轴为法向量方向。
    主要用于抗扭矩计算中的坐标系变换。
    
    参数:
        towards (numpy.ndarray): 法向量，形状为(3,)
    
    返回:
        numpy.ndarray: 3x3旋转矩阵，z轴对齐法向量方向
    """
    n = towards
    # 设置z轴为法向量方向
    new_z = n
    # 构造y轴，垂直于z轴在水平面内
    new_y = np.array((new_z[1], -new_z[0], 0), dtype=np.float64)
    new_y = new_y / np.linalg.norm(new_y)
    # 标准化z轴
    new_z = new_z / np.linalg.norm(new_z)
    # 通过叉积计算x轴
    new_x = np.cross(new_y, new_z)
    new_x = new_x / np.linalg.norm(new_x)
    # 扩展维度并组合成旋转矩阵
    new_x = np.expand_dims(new_x, axis=1)
    new_y = np.expand_dims(new_y, axis=1)
    new_z = np.expand_dims(new_z, axis=1)  
    rot_matrix = np.concatenate((new_x, new_y, new_z), axis=-1)
    return rot_matrix

class H5DataGenerator(object):
    """
    H5数据集生成器类
    
    该类是整个数据处理管道的核心，负责将多模态原始数据转换为
    标准化的H5格式训练数据集。集成了点云处理、法向量估计、
    多维度评分计算等功能。
    """
    
    def __init__(self, params_file_name, camera_info_file_name, objs_path, target_num_point=16384, test_flag=False):
        """
        初始化数据生成器，加载相机参数和处理配置
        
        参数:
            params_file_name (str): 参数配置文件路径("parameter.json")
            target_num_point (int): 点云采样的目标点数，默认16384
                                   这个数量平衡了计算效率和数据质量
            enable_visualization (bool): 是否启用可视化，WSL环境中建议设为False
        """
        # 加载深度范围等关键参数
        self.params = self._load_parameters(params_file_name)
        # 加载相机参数
        self.cam_info = CameraInfo(camera_info_file_name)
        # 设置点云采样目标数量，确保数据一致性
        self.target_num_point = target_num_point
        self.objs_path = objs_path
        # 可视化开关，WSL环境中建议关闭
        self.test_flag = test_flag

    def _depth_to_pointcloud_optimized(self, us, vs, zs, to_mm=False):
        """
        将深度图像像素坐标转换为3D点云坐标
        
        使用相机内参进行投影变换，从2D像素坐标和深度值重建3D空间坐标。
        这是整个数据处理管道的基础步骤。
        
        参数:
            us (numpy.ndarray): u坐标数组（像素水平坐标）
            vs (numpy.ndarray): v坐标数组（像素垂直坐标）  
            zs (numpy.ndarray): 深度值数组（归一化深度）
            to_mm (bool): 是否转换为毫米单位，默认False（米单位）
            xyz_limit (list): 3D空间裁剪范围，格式[[xmin,xmax], [ymin,ymax], [zmin,zmax]]
                             用于过滤工作空间外的点
        
        返回:
            numpy.ndarray: 3D点云坐标，形状为(N, 3)
        备注：
            blender中相机朝向为Z轴的负方向，因此在相机坐标系中深度应该为负值
        """
        assert len(us) == len(vs) == len(zs), "坐标数组长度必须一致"
        
        # 从参数配置中获取相机内参
        fx = self.cam_info.intrinsic_matrix[0, 0]
        fy = self.cam_info.intrinsic_matrix[1, 1]
        cx = self.cam_info.intrinsic_matrix[0, 2]  # x方向主点坐标
        cy = self.cam_info.intrinsic_matrix[1, 2]  # y方向主点坐标
        clip_start = self.params['clip_start']  # 近裁剪面距离
        clip_end = self.params['clip_end']      # 远裁剪面距离
        
        # 将归一化深度值转换为真实距离（米）
        # 深度图中的值通常是归一化的，需要映射到真实距离范围
        Zline = clip_start + (zs/self.params['max_val_in_depth']) * (clip_end - clip_start)
        
        # 考虑透视投影的距离校正
        # 校正由于透视投影导致的距离失真
        Zcs = Zline/np.sqrt(1+ np.power((us-cx)/fx,2) + np.power((vs-cy)/fy,2))
        
        # 可选：转换为毫米单位（某些应用需要）
        if to_mm:
            Zcs *= 1000
            
        # 使用针孔相机模型进行3D重建
        # X = (u - cx) * Z / fx, Y = (v - cy) * Z / fy
        Xcs = (us - cx) * Zcs / fx
        Ycs = (vs - cy) * Zcs / fy
        
        # 重塑为列向量并组合成点云
        Xcs = np.reshape(Xcs, (-1, 1)) # 列向量
        Ycs = np.reshape(Ycs, (-1, 1))
        Zcs = np.reshape(Zcs, (-1, 1))
        # blender中相机朝向为Z轴的负方向，因此在相机坐标系中的坐标应该取反
        points = np.concatenate([Xcs, -Ycs, -Zcs], axis=-1)
                    
        return points

    def _load_parameters(self, params_file_name):
        """
        加载相机参数和处理配置文件
        
        参数:
            params_file_name (str): JSON配置文件路径
            
        返回:
            dict: 包含相机内参、深度范围等配置的字典
        """
        params = {}
        with open(params_file_name, 'r') as f:
            config = json.load(f)
            params = config
        return params 

    def _read_label_csv(self, file_name):
        """
        读取场景真值标签CSV文件，获取物体位姿信息
        
        解析CSV文件中的物体6D位姿数据（位置+旋转矩阵），
        这些数据来自物理仿真或标注工具。
        
        参数:
            file_name (str): 真值CSV文件路径
            
        返回:
            tuple: (平移向量, 旋转矩阵, 物体ID, 物体名称)
                - label_trans: (N, 3) 物体位置坐标
                - label_rot: (N, 9) 旋转矩阵（展平为9维）
                - label_id: (N, 1) 物体唯一标识符
                - label_name: (N,) 物体名称列表
        """
        with open(file_name, 'r') as csv_file:  
            all_lines = csv.reader(csv_file) 
            list_file = [i for i in all_lines]  
        
        # 排除CSV表头，获取数据行
        array_file = np.array(list_file)[1:]
        num_obj = int(array_file.shape[0])
        
        # 解析各列数据：位置(3列) + 旋转矩阵(9列) + ID(1列) + 名称(1列)
        label_trans = array_file[:, 2:5].astype('float32')    # 物体xyz位置
        label_rot = array_file[:, 5:14].astype('float32')     # 3x3旋转矩阵展平
        label_id = array_file[:, 1:2].astype('float32')       # 物体编号
        label_name = array_file[:, 0]                         # 物体名称
        
        return label_trans, label_rot, label_id, label_name

    def individual_label_csv(self, file_name):
        """
        读取单个物体的尺寸标签CSV文件
        
        该文件包含每个物体的可见面积比例信息，用于评估
        吸取任务中物体的暴露程度和可操作性。
        
        参数:
            file_name (str): 物体尺寸标签CSV文件路径
            
        返回:
            numpy.ndarray: 物体尺寸标签数组，数据类型为float32
        """
        with open(file_name, 'r') as csv_file:  
            all_lines = csv.reader(csv_file) 
            list_file = [i for i in all_lines]  
        # 直接转换为float32数组，不排除标题行（因为数据文件格式）
        array_file = np.array(list_file).astype('float32')
        return array_file

    def create_mesh_cylinder(self, radius, height, R, t, collision):
        """
        创建用于可视化的圆柱体网格
        
        生成圆柱体几何体用于可视化吸取点的法向量方向和评分，
        颜色编码表示吸取质量或碰撞状态。
        
        参数:
            radius (float): 圆柱体半径
            height (float): 圆柱体高度
            R (numpy.ndarray): 3x3旋转矩阵，定义圆柱体朝向
            t (numpy.ndarray): 3D平移向量，定义圆柱体位置
            collision (float): 碰撞评分，用于颜色编码（0-1范围）
            
        返回:
            open3d.geometry.TriangleMesh: 带颜色的圆柱体网格
        """
        # 创建标准圆柱体（底面在原点，向上延伸）
        cylinder = o3d.geometry.TriangleMesh().create_cylinder(radius, height)
        vertices = np.asarray(cylinder.vertices)
        
        # 将圆柱体底面移到原点上方，使其以底面中心为旋转原点
        vertices[:, 2] += height / 2
        
        # 应用旋转和平移变换
        vertices = np.dot(R, vertices.T).T + t
        cylinder.vertices = o3d.utility.Vector3dVector(vertices)
        
        # 根据碰撞评分设置颜色：红色分量表示碰撞程度
        ball_colors = np.zeros((vertices.shape[0], 3), dtype=np.float32)
        ball_colors[:, 0] = collision  # 红色通道，值越大越红
        cylinder.vertex_colors = o3d.utility.Vector3dVector(ball_colors)
        
        return cylinder
    def _cal_score_seal(self, suction_points, obj_ids, label_trans, label_rot, label_name):
        '''
        计算密封评分，密封评分用于确定在以特定姿势进行吸力抓取时吸盘是否能保持真空状态
        '''

        actual_num_points = suction_points.shape[0]
        
        # 坐标归一化到物体坐标系 - 使用实际点数
        suction_points_normalization = np.matmul(
            (suction_points - label_trans).reshape(actual_num_points, 1, 3), 
            label_rot.reshape(-1, 3, 3)
        )
        suction_points_normalization = suction_points_normalization.reshape(actual_num_points, 3)
        
        # 初始化评分数组 - 使用实际点数
        suction_seal_scores = np.zeros((actual_num_points,))
        
        # 计算吸取点的密封分数
        for index in range(len(label_name)):
            # 读取每个物体的稀疏点和分数
            try:
                annotation = np.load(os.path.join(self.objs_path, label_name[index], "labels.npz"))
                object_sparse_point = annotation['points']
                anno_points = annotation['points']
                anno_scores = annotation['scores']
            except FileNotFoundError:
                print(f"警告: 找不到物体 {label_name[index]} 的标签文件")
                continue
            except KeyError as e:
                print(f"警告: 物体 {label_name[index]} 的标签文件缺少键: {e}")
                continue

            suction_points_normalization_id = suction_points_normalization[obj_ids == index]
            if suction_points_normalization_id.shape[0] == 0:
                print(f"物体 {index} ({label_name[index]}) 没有对应的点")
                continue
                
            print(f"处理物体 {index} ({label_name[index]}): {suction_points_normalization_id.shape[0]} 个点")
            
            suction_points_normalization_id_knn = torch.from_numpy(suction_points_normalization_id).float()
            anno_points_knn = torch.from_numpy(anno_points).float()
            
            # 优先用GPU，失败则自动切换CPU
            try:
                suction_points_normalization_id_knn = suction_points_normalization_id_knn.cuda()
                anno_points_knn = anno_points_knn.cuda()
                indices, dist = knn(anno_points_knn, suction_points_normalization_id_knn, k=1)
                dist = dist.cpu().numpy().reshape(dist.shape[-1])
            except RuntimeError as e:
                print(f"KNN CUDA失败，切换到CPU: {e}")
                suction_points_normalization_id_knn = suction_points_normalization_id_knn.cpu()
                anno_points_knn = anno_points_knn.cpu()
                indices, dist = knn(anno_points_knn, suction_points_normalization_id_knn, k=1)
                dist = dist.numpy().reshape(dist.shape[-1])
            suction_seal_scores[obj_ids == index] = anno_scores[dist]
        return suction_seal_scores
    
    def _score_seel_visiualization(self, score_seal, suction_points, suction_or):
        # 可视化吸取分数
        show_point_temp=o3d.geometry.PointCloud(o3d.utility.Vector3dVector(suction_points))
        colors_temp = [[0, 0, 1]  for i in range(suction_points.shape[0])]
        show_point_temp.colors = o3d.utility.Vector3dVector(colors_temp)
        vis_list = [  show_point_temp   ]
        for idx in range(len(suction_points[0:1024*4])):
            suction_point = suction_points[idx]
            suction_score = score_seal[idx]
            ball = o3d.geometry.TriangleMesh.create_sphere(0.001).translate(suction_point)
            ball_v = np.asarray(ball.vertices)
            ball_colors = np.zeros((ball_v.shape[0], 3), dtype=np.float32)
            ball_colors[:, 0] = suction_score
            ball.vertex_colors = o3d.utility.Vector3dVector(ball_colors)
            vis_list.append(ball)
        # 可视化前100个吸取点的法线和分数
        for idx in range(len(suction_points[0:100])):
            suction_point = suction_points[idx]
            anno_normal = suction_or[idx]
            suction_score = score_seal[idx]
            n = anno_normal
            new_z = n
            new_y = np.array((new_z[1], -new_z[0], 0), dtype=np.float64)
            new_y = new_y / np.linalg.norm(new_y)
            new_x = np.cross(new_y, new_z)
            new_x = new_x / np.linalg.norm(new_x)
            new_x = np.expand_dims(new_x, axis=1)
            new_y = np.expand_dims(new_y, axis=1)
            new_z = np.expand_dims(new_z, axis=1)
            rot_matrix = np.concatenate((new_x, new_y, new_z), axis=-1)
            ball = self.create_mesh_cylinder(radius=0.005, height=0.05, R=rot_matrix, t=suction_point, collision=suction_score)
            vis_list.append(ball)
        o3d.visualization.draw_geometries(vis_list, width=800, height=600)
        # 绘制吸取分数直方图
        plt.hist(score_seal, bins=100)
        plt.title("Histogram of SealScore")
        plt.xlabel("Value")
        plt.ylabel("Frequency")
        plt.show()
        return
    
    def _cal_score_wrench(self, suction_points, suction_or, label_trans, camera_info):
        '''
        计算抗扭矩评分，抗扭矩评分用于判断吸盘在特定姿势下是否无法抵抗重力
        '''
        # 计算吸取点的抗扭分数(考虑重力和吸盘姿态)
        # 原算法
        # k = 30
        # radius = 0.01
        # wrench_thre = k * radius * np.pi * np.sqrt(2)
        # suction_wrench_scores = []
        # for index_temp,suction_points_temp in enumerate(suction_points):
        #     label_trans_temp = label_trans[index_temp]
        #     suction_or_temp = suction_or[index_temp]
        #     center = label_trans_temp
        #     gravity = np.array([[0, 0, 1]], dtype=np.float32) * 9.8  # 重力方向
        #     suction_axis = viewpoint_to_matrix_z(suction_or_temp)  # (3, 3)
        #     suction2center = (center - suction_points_temp)[np.newaxis, :]
        #     coord = np.matmul(suction2center, suction_axis)
        #     gravity_proj = np.matmul(gravity, suction_axis)
        #     torque_y = gravity_proj[0, 0] * coord[0, 2] - gravity_proj[0, 2] * coord[0, 0]
        #     torque_x = -gravity_proj[0, 1] * coord[0, 2] + gravity_proj[0, 2] * coord[0, 1]
        #     torque = np.sqrt(torque_x**2 + torque_y**2)
        #     score = 1 - min(1, torque / wrench_thre)
        #     suction_wrench_scores.append(score)
        # suction_wrench_scores = np.array(suction_wrench_scores)
        # 新算法
        k = 30
        radius = 0.01
        wrench_thre = k * radius * np.pi * np.sqrt(2)
        suction_wrench_scores = []
        for index_temp,suction_points_temp in enumerate(suction_points):
            # index_temp：当前点云中点的索引
            # suction_points_temp：当前点云中点的坐标
            # 获取点所属物体的中心
            label_trans_temp = label_trans[index_temp]
            # 获取点对应的法向量
            suction_or_temp = suction_or[index_temp]
            # 重力作用点
            center = label_trans_temp
            # 重力方向
            gravity = np.array([[0, 0, -1]], dtype=np.float32) * 9.8  # 重力方向
            # 获取点对应的法向量对应的旋转矩阵
            suction_axis = viewpoint_to_matrix_z(suction_or_temp)  # (3, 3)
            # 计算力臂（从吸取点到物体中心的向量）
            suction2center = (center - suction_points_temp)[np.newaxis, :]
            # 将力臂向量从世界坐标系转换到吸盘局部坐标系
            coord = np.matmul(suction2center, suction_axis)

            # 将重力向量投影到吸盘局部坐标系
            gravity_proj = np.matmul(gravity, suction_axis)
            # 计算吸盘坐标系下重力的扭矩
            # 如果 G = [Gx, Gy, Gz]，F = [Fx, Fy, Fz]
            # 那么 τ = F x G  = [Gy * Fz - Gz * Fy, Gz * Fx - Gx * Fz, Gx * Fy - Gy * Fx]
            torque_y = gravity_proj[0, 0] * coord[0, 2] - gravity_proj[0, 2] * coord[0, 0]
            torque_x = -gravity_proj[0, 1] * coord[0, 2] + gravity_proj[0, 2] * coord[0, 1]
            torque = np.sqrt(torque_x**2 + torque_y**2)
            score = 1 - min(1, torque / wrench_thre)
            suction_wrench_scores.append(score)
        # 确保初始转换为float64类型，避免后续计算中的数据类型问题
        suction_wrench_scores = np.array(suction_wrench_scores, dtype=np.float64)
        
        # 定义参考向量，指向负Z轴方向（垂直向下，符合重力方向）
        reference_vector = np.array([0, 0, -1])

        # 计算法向量与参考向量的点积（向量内积）
        dot_products = np.sum(suction_or * reference_vector, axis=1)

        # 计算每个法向量的模长（向量长度）
        norm_magnitudes = np.linalg.norm(suction_or, axis=1)
        # 计算参考向量的模长
        reference_magnitude = np.linalg.norm(reference_vector)

        # 根据向量点积公式计算夹角的余弦值: cos(θ) = (a·b) / (|a|*|b|)
        cos_angles = dot_products / (norm_magnitudes * reference_magnitude)

        # 防止数值计算误差导致余弦值超出[-1, 1]范围
        cos_angles = np.clip(cos_angles, -1.0, 1.0)

        # 通过反余弦函数计算夹角（弧度制）
        angles = np.arccos(cos_angles)

        # 将弧度转换为角度制，便于理解和调试
        angles_degrees = np.degrees(angles)

        # 将角度映射到[0,1]权重区间：
        # - 0度（完全垂直向下）→ 权重1.0
        # - 90度（水平方向）→ 权重0.5  
        # - 180度（完全垂直向上）→ 权重0.0
        mapped_values = 1 - (angles_degrees / 180)

        # 确保数据类型兼容，避免casting错误
        suction_wrench_scores = suction_wrench_scores.astype(np.float64)
        mapped_values = mapped_values.astype(np.float64)
        suction_wrench_scores *= mapped_values # 乘以权重
        return suction_wrench_scores
    
    def _cal_score_collision(self, suction_points, suction_or):
        '''
        计算碰撞评分，碰撞评分用于评估吸取点与其他物体的几何碰撞情况
        '''
        # 计算吸取点的可行性分数(碰撞检测)
        height = 0.15
        radius = 0.02
        scence_point = suction_points
        suction_feasibility_scores = []
        for index_temp,suction_points_temp in enumerate(suction_points):
            suction_or_temp = suction_or[index_temp]
            grasp_poses = viewpoint_to_matrix_x(suction_or_temp)
            target = scence_point-suction_points_temp
            target = np.matmul(target, grasp_poses)
            target_yz = target[:, 1:3]
            target_r = np.linalg.norm(target_yz, axis=-1)
            mask1 = target_r < radius
            mask2 = ((target[:,0] > 0.01) & (target[:,0] < height))
            mask = np.any(mask1 & mask2)
            suction_feasibility_scores.append(mask)
        suction_feasibility_scores = ~np.array(suction_feasibility_scores)
        return suction_feasibility_scores
    
    def _cal_score_visibility(self, visibility_label, obj_ids):
        '''
        计算可见性评分，可见性评分用于定量反映场景中对象的被遮挡程度
        '''
        suction_visibility_scores = np.array([visibility_label[obj_ids[i]] for i in obj_ids])
        return suction_visibility_scores
    
    def _filter_points(self, points):
        """
        使用半径滤波去除点云中的离群点
        
        半径滤波的原理：
        1. 对每个点，在指定半径内搜索邻居点
        2. 如果邻居点数量少于指定阈值，则认为该点是离群点
        3. 移除所有离群点
        
        参数:
            points (numpy.ndarray): 输入点云，形状为(N, 3)
            
        返回:
            tuple: (过滤后的点云, 法向量, 有效点的索引掩码)
        """
        try:
            # 创建Open3D点云对象
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(points)
            
            # 应用半径滤波
            pcd_filtered, inlier_indices = pcd.remove_radius_outlier(
                nb_points=16, radius=0.01
            )
            
            # 转换回numpy数组
            filtered_points = np.asarray(pcd_filtered.points)
            
            # 创建正确的索引掩码
            inlier_mask = np.zeros(points.shape[0], dtype=bool)
            inlier_mask[inlier_indices] = True
            
            print(f"半径滤波: 原始点数 {points.shape[0]}, 过滤后点数 {filtered_points.shape[0]}")
            print(f"掩码统计: True={np.sum(inlier_mask)}, False={np.sum(~inlier_mask)}")
            
            # 计算法向量（只对过滤后的点）
            if filtered_points.shape[0] > 0:
                pc_o3d = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(filtered_points))
                pc_o3d.estimate_normals(
                    o3d.geometry.KDTreeSearchParamRadius(0.015), 
                    fast_normal_computation=False
                )
                
                # # 计算相机坐标系下地球方向并统一法向量
                # world_down = np.array([0., 0., -1.])
                # w2c_rotation = self.cam_info.extrinsic_matrix[:3, :3]
                # cam_down = np.matmul(w2c_rotation, world_down)
                # pc_o3d.orient_normals_to_align_with_direction(cam_down)
                pc_o3d.normalize_normals()
                
                suction_or = np.array(pc_o3d.normals).astype(np.float32)
            else:
                suction_or = np.array([]).reshape(0, 3)
            
            return filtered_points, suction_or, inlier_mask
            
        except Exception as e:
            print(f"半径滤波失败，使用原始点云: {e}")
            identity_mask = np.ones(points.shape[0], dtype=bool)
            dummy_normals = np.zeros((points.shape[0], 3), dtype=np.float32)
            dummy_normals[:, 2] = -1
            return points, dummy_normals, identity_mask
                
    
    def _transform_points(self, points):
        """
        将点云从相机坐标系转换到世界坐标系
        
        使用4x4齐次变换矩阵进行完整的刚体变换（旋转+平移）
        
        参数:
            points (numpy.ndarray): 相机坐标系下的点云，形状为(N, 3)
            
        返回:
            numpy.ndarray: 世界坐标系下的点云，形状为(N, 3)
        """
        # 将3D点扩展为齐次坐标（添加第4维度为1）
        ones = np.ones((points.shape[0], 1))
        points_homo = np.hstack([points, ones])  # 形状: (N, 4)
        
        #外参矩阵为W2C矩阵即世界坐标系转相机坐标系的矩阵，因此需要先计算C2W
        c2w = np.linalg.inv(self.cam_info.extrinsic_matrix)
        
        # 使用4x4外参矩阵进行变换
        # 外参矩阵将相机坐标系转换为世界坐标系
        points_world_homo = c2w @ points_homo.T # 形状: (4, N)
        
        # 提取前3个维度，去除齐次坐标
        points_world = (points_world_homo.T)[:, :3] 
        
        return points_world # 形状: (N, 3)
    
    def _transform_normals(self, normals):
        """
        将法向量从相机坐标系转换到世界坐标系
        
        法向量只需要旋转变换，不需要平移
        
        参数:
            normals (numpy.ndarray): 相机坐标系下的法向量，形状为(N, 3)
            
        返回:
            numpy.ndarray: 世界坐标系下的法向量，形状为(N, 3)
        """
        # 法向量只需要旋转变换，提取旋转矩阵部分
        rotation_matrix = self.cam_info.extrinsic_matrix[:3, :3].T
        
        print(f'normals.shape : {normals.shape}')
        # 应用旋转变换
        normals_world = (rotation_matrix @ normals.T).T # 形状: (N, 3)
        
        # 重新归一化法向量
        norms = np.linalg.norm(normals_world, axis=1, keepdims=True)
        normals_world = normals_world / (norms + 1e-8)  # 避免除零
        
        return normals_world
    
    def _filter_z_axis(self, points, normals, prev_filter_mask, z_threshold = 0.7):
        """
        根据Z轴坐标过滤点云，剔除Z轴坐标大于阈值的点
        
        参数:
            points (numpy.ndarray): 输入点云，形状为(N, 3)
            normals (numpy.ndarray): 对应的法向量，形状为(N, 3)
            prev_filter_mask (numpy.ndarray): 之前的过滤掩码，用于累积过滤效果
            
        返回:
            tuple: (过滤后的点云, 过滤后的法向量, 更新后的过滤掩码)
        """
        
        # 创建Z轴过滤掩码：保留Z坐标小于等于阈值的点
        z_filter_mask = points[:, 2] <= z_threshold
        
        # 过滤点云和法向量
        filtered_points = points[z_filter_mask]
        filtered_normals = normals[z_filter_mask]
        
        # 更新累积过滤掩码
        # 创建新的累积掩码，标记哪些原始点被保留
        new_filter_mask = np.zeros_like(prev_filter_mask, dtype=bool)
        
        # 在之前被保留的点中，进一步标记通过Z轴过滤的点
        prev_indices = np.where(prev_filter_mask)[0]
        z_passed_indices = prev_indices[z_filter_mask]
        new_filter_mask[z_passed_indices] = True
        
        print(f"Z轴过滤: 输入点数 {points.shape[0]}, 过滤后点数 {filtered_points.shape[0]}, "
              f"移除 {points.shape[0] - filtered_points.shape[0]} 个Z>{z_threshold}m的点")
        
        return filtered_points, filtered_normals, new_filter_mask
    
    def _visualize_pointcloud(self, points):
        """
        使用matplotlib可视化点云和法向量
        
        参数:
            points (numpy.ndarray): 点云数据，形状为(N, 3)
            normals (numpy.ndarray): 法向量数据，形状为(N, 3)
        """
        # 检测是否在WSL或非图形环境中运行

        
        # 检查是否在主线程中
        is_main_thread = threading.current_thread() is threading.main_thread()
        
        # 检查是否有显示环境
        has_display = 'DISPLAY' in os.environ or 'WAYLAND_DISPLAY' in os.environ
        
        # 如果不在主线程或没有显示环境，使用非交互式后端
        if not is_main_thread or not has_display:
            matplotlib.use('Agg')  # 使用非交互式后端
            print("检测到非主线程或无显示环境，直接保存点云文件")
            
        
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        
        # 直接使用点云的Z轴坐标值进行颜色渲染
        # 将Z轴值归一化到0-1范围用于颜色映射
        z_values = points[:, 2]
        z_min, z_max = z_values.min(), z_values.max()
        if z_max > z_min:
            colors = (z_values - z_min) / (z_max - z_min)  # 归一化到[0,1]
        else:
            colors = np.zeros_like(z_values)  # 如果Z值都相同，使用单一颜色
        
        ax.scatter(points[:, 0], points[:, 1], points[:, 2], c=colors, s=5, cmap='viridis')
        # 添加坐标轴
        # 绘制坐标轴箭头
        max_range = np.array([points[:, 0].max()-points[:, 0].min(), 
                            points[:, 1].max()-points[:, 1].min(),
                            points[:, 2].max()-points[:, 2].min()]).max() / 2.0
        
        mid_x = (points[:, 0].max()+points[:, 0].min()) * 0.5
        mid_y = (points[:, 1].max()+points[:, 1].min()) * 0.5
        mid_z = (points[:, 2].max()+points[:, 2].min()) * 0.5
        
        # 绘制坐标轴
        axis_length = max_range * 0.5
        ax.quiver(mid_x, mid_y, mid_z, axis_length, 0, 0, color='red', arrow_length_ratio=0.1)
        ax.quiver(mid_x, mid_y, mid_z, 0, axis_length, 0, color='green', arrow_length_ratio=0.1)
        ax.quiver(mid_x, mid_y, mid_z, 0, 0, axis_length, color='blue', arrow_length_ratio=0.1)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title('Point Cloud Visualization')
        
        if not is_main_thread or not has_display:
            # 保存图像到文件而不是显示
            output_path = '/tmp/pointcloud_visualization.png'
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            print(f"点云可视化已保存到: {output_path}")
            plt.close(fig)  # 关闭图形以释放内存
        else:
            plt.show()
        
    def _normalize_pointcloud(self, points, obj_ids, translations):
        """
        点云标准化处理流程
        
        包含以下步骤：
        1. 统计滤波去除离群点
        2. 坐标系转换到世界坐标系
        
        参数:
            points (numpy.ndarray): 输入原始点云
            
        返回:
            tuple: (标准化后的点云, 有效点的索引掩码)
        """
        # 第1步：滤波
        filtered_points, camera_normals, filter_mask = self._filter_points(points)
    
        # 第2步：坐标转换到世界坐标系
        transformed_points = self._transform_points(filtered_points)
        normals = self._transform_normals(camera_normals)
        filter_obj_ids = obj_ids[filter_mask]
        # 向量全部指向各个物体中心,mask记录滤波后的法向量中有哪些被翻转了
        fliped_normals, mask = self._normalize_normals(transformed_points, normals, filter_obj_ids, translations)
        # # 第3步：剔除Z轴坐标大于0.7的点
        transformed_points, normals, filter_mask = self._filter_z_axis(transformed_points, normals, filter_mask)
        
        return transformed_points, normals, fliped_normals, filter_mask, mask
    
    def _process_obj_normals(self, points, normals, obj_center):
        # 计算从点指向中心的向量（真正的"点到中心"向量）
        points_to_center = obj_center - points  # 形状为(N, 3)
        
        # 计算法向量与指向中心向量的点积
        dot_products = np.sum(normals * points_to_center, axis=1)  # 形状 (N,)

        # 找到反向的点（点积小于0，即法向量背离中心）
        opposite_direction_mask = dot_products < 0  # 形状 (N,)

        # 对背离中心的法向量取反，使其指向中心
        normals[opposite_direction_mask] *= -1
        # self._visualize_normals_and_center(points, normals, obj_center)
        return normals, opposite_direction_mask
    
    def _visualize_normals_and_center(self, points, normals, obj_center, max_normals=50):
        """
        可视化点云、法向量和物体中心点
        
        参数:
            points (numpy.ndarray): 点云数据，形状为(N, 3)
            normals (numpy.ndarray): 法向量数据，形状为(N, 3)
            obj_center (numpy.ndarray): 物体中心点，形状为(3,)
            max_normals (int): 最大显示的法向量数量，避免过于密集
        """
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D
        import threading
        
        # 检查是否在主线程和有显示环境
        is_main_thread = threading.current_thread() is threading.main_thread()
        has_display = 'DISPLAY' in os.environ or 'WAYLAND_DISPLAY' in os.environ
        
        if not is_main_thread or not has_display:
            matplotlib.use('Agg')
            print("检测到非主线程或无显示环境，保存可视化图片")
        
        fig = plt.figure(figsize=(14, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        # 1. 绘制点云
        ax.scatter(points[:, 0], points[:, 1], points[:, 2], 
                c='lightblue', s=8, alpha=0.7, label='Point Cloud')
        
        # 2. 绘制物体中心点
        ax.scatter(obj_center[0], obj_center[1], obj_center[2], 
                c='red', s=200, marker='*', 
                label='Object Center', edgecolors='darkred', linewidth=2)
        
        # 3. 绘制法向量（选择性显示，避免过于密集）
        num_points = len(points)
        if num_points > max_normals:
            # 均匀采样法向量索引
            step = num_points // max_normals
            indices = np.arange(0, num_points, step)[:max_normals]
        else:
            indices = np.arange(num_points)
        
        # 计算法向量的缩放长度（基于点云尺度）
        points_range = np.max(points, axis=0) - np.min(points, axis=0)
        normal_length = np.mean(points_range) * 0.05  # 法向量长度为点云尺度的5%
        
        # 绘制选中的法向量
        for i in indices:
            point = points[i]
            normal = normals[i] * normal_length
            
            # 绘制法向量箭头
            ax.quiver(point[0], point[1], point[2],
                    normal[0], normal[1], normal[2],
                    color='yellow', alpha=0.8, arrow_length_ratio=0.1)
        
        # 4. 绘制从点到中心的连线（选择几个代表性的点）
        representative_indices = indices[::max(1, len(indices)//10)]  # 选择最多10条连线
        for i in representative_indices:
            point = points[i]
            ax.plot([point[0], obj_center[0]], 
                    [point[1], obj_center[1]], 
                    [point[2], obj_center[2]], 
                    'g--', alpha=0.5, linewidth=1)
        
        # 5. 设置坐标轴等比例和标签
        # 计算显示范围
        all_points = np.vstack([points, obj_center.reshape(1, -1)])
        max_range = np.array([all_points[:, 0].max() - all_points[:, 0].min(),
                            all_points[:, 1].max() - all_points[:, 1].min(),
                            all_points[:, 2].max() - all_points[:, 2].min()]).max() / 2.0
        
        mid_x = (all_points[:, 0].max() + all_points[:, 0].min()) * 0.5
        mid_y = (all_points[:, 1].max() + all_points[:, 1].min()) * 0.5
        mid_z = (all_points[:, 2].max() + all_points[:, 2].min()) * 0.5
        
        # 设置等比例显示
        expansion_factor = 1.2  # 稍微扩大显示范围
        ax.set_xlim(mid_x - max_range * expansion_factor, mid_x + max_range * expansion_factor)
        ax.set_ylim(mid_y - max_range * expansion_factor, mid_y + max_range * expansion_factor)
        ax.set_zlim(mid_z - max_range * expansion_factor, mid_z + max_range * expansion_factor)
        
        # 6. 添加坐标轴
        axis_length = max_range * 0.3
        ax.quiver(mid_x, mid_y, mid_z, axis_length, 0, 0, 
                color='red', arrow_length_ratio=0.1, linewidth=3, alpha=0.8)
        ax.quiver(mid_x, mid_y, mid_z, 0, axis_length, 0, 
                color='green', arrow_length_ratio=0.1, linewidth=3, alpha=0.8)
        ax.quiver(mid_x, mid_y, mid_z, 0, 0, axis_length, 
                color='blue', arrow_length_ratio=0.1, linewidth=3, alpha=0.8)
        
        # 添加坐标轴标签
        ax.text(mid_x + axis_length * 1.1, mid_y, mid_z, 'X', 
                color='red', fontsize=12, fontweight='bold')
        ax.text(mid_x, mid_y + axis_length * 1.1, mid_z, 'Y', 
                color='green', fontsize=12, fontweight='bold')
        ax.text(mid_x, mid_y, mid_z + axis_length * 1.1, 'Z', 
                color='blue', fontsize=12, fontweight='bold')
        
        # 7. 设置标题和标签
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_zlabel('Z (m)')
        ax.set_title(f'Object Normal Vectors Alignment\n'
                    f'Points: {len(points)}, Normals shown: {len(indices)}')
        
        # 8. 添加图例
        ax.legend(loc='upper left', bbox_to_anchor=(0, 1))
        
        # 9. 添加统计信息文本
        # 验证法向量指向中心的比例
        points_to_center = obj_center - points
        dot_products = np.sum(normals * points_to_center, axis=1)
        pointing_to_center = np.sum(dot_products > 0)
        pointing_percentage = pointing_to_center / len(points) * 100
        
        info_text = f"Statistics:\n" \
                    f"Total points: {len(points)}\n" \
                    f"Pointing to center: {pointing_to_center}/{len(points)}\n" \
                    f"Alignment rate: {pointing_percentage:.1f}%"
        
        ax.text2D(0.02, 0.98, info_text, transform=ax.transAxes, 
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        # 10. 保存或显示
        plt.tight_layout()
        
        if not is_main_thread or not has_display:
            output_path = 'normals_alignment_visualization.png'
            plt.savefig(output_path, dpi=200, bbox_inches='tight', 
                    facecolor='white', edgecolor='none')
            print(f"法向量对齐可视化已保存到: {output_path}")
            plt.close(fig)
        else:
            plt.show()
        
        # 11. 输出详细统计信息
        print(f"\n=== 法向量对齐统计 ===")
        print(f"物体中心坐标: [{obj_center[0]:.3f}, {obj_center[1]:.3f}, {obj_center[2]:.3f}]")
        print(f"点云数量: {len(points)}")
        print(f"指向中心的法向量: {pointing_to_center}/{len(points)} ({pointing_percentage:.1f}%)")
        print(f"背离中心的法向量: {len(points) - pointing_to_center}/{len(points)} ({100 - pointing_percentage:.1f}%)")
        
        # 计算平均角度
        norm_normals = np.linalg.norm(normals, axis=1)
        norm_to_center = np.linalg.norm(points_to_center, axis=1)
        cos_angles = dot_products / (norm_normals * norm_to_center + 1e-8)
        cos_angles = np.clip(cos_angles, -1.0, 1.0)
        angles_deg = np.degrees(np.arccos(np.abs(cos_angles)))
        
        print(f"平均偏离角度: {np.mean(angles_deg):.2f}°")
        print(f"最大偏离角度: {np.max(angles_deg):.2f}°")
        print(f"最小偏离角度: {np.min(angles_deg):.2f}°")
        print("=" * 30)
    def _normalize_normals(self, points, normals, obj_ids, translations):
        # 获取唯一的物体ID
        print(f"points.shape: {points.shape}")
        print(f"normals.shape: {normals.shape}")
        print(f"obj_ids.shape: {obj_ids.shape}")
        print(f"translations.shape: {translations.shape}")
        unique_ids = np.unique(obj_ids)
        
        # 存储分离后的点云
        separated_clouds = {}
        # 存储计算后的向量
        normals_normalized = np.zeros_like(normals, dtype=np.float32)
        all_mask = np.zeros(points.shape[0], dtype=bool)
        for obj_id in unique_ids:
            # 创建布尔掩码，选择属于当前物体的点
            mask = (obj_ids == obj_id)
            # 计算indices
            
            # 提取对应的数据
            obj_data = {
                'points': points[mask],
                'normals': normals[mask],
                'obj_id': obj_id,# 场景中物体索引非物体ID
                'num_points': np.sum(mask),
                'translation': translations[obj_id, :],  # 获取物体的平移向量
            }
            
            separated_clouds[int(obj_id)] = obj_data
            single_points = obj_data['points']
            single_normals = obj_data['normals']
            center = obj_data['translation']
            print(f"处理当前场景中第 {obj_id} 个物体，点数: {single_points.shape[0]}, 中心: {center}")
            normals_normalized[mask], all_mask[mask] = self._process_obj_normals(single_points, single_normals, center)
        return normals_normalized, all_mask
    def _visualize_pointcloud_with_ids(self, points, obj_ids, point_size=1):
        """
        使用matplotlib可视化点云, 根据点所属物体ID进行上色
        支持50+个物体的颜色区分
        
        参数:
            points (numpy.ndarray): 点云数据，形状为(N, 3)
            obj_ids (numpy.ndarray): 对应的物体ID，形状为(N,)
            point_size (float): 点的大小，默认为1
        """
        # from matplotlib.colors import ListedColormap

        # 检查是否在主线程和有显示环境
        is_main_thread = threading.current_thread() is threading.main_thread()
        has_display = 'DISPLAY' in os.environ or 'WAYLAND_DISPLAY' in os.environ

        # WSL或无显示环境下使用非交互式后端
        if not is_main_thread or not has_display:
            matplotlib.use('Agg')
            print("检测到WSL或无显示环境，自动保存点云可视化图片")

        fig = plt.figure(figsize=(14, 10))  # 增大图形尺寸
        ax = fig.add_subplot(111, projection='3d')

        # 物体ID着色 - 支持50+个物体的丰富色系
        unique_ids = np.unique(obj_ids)
        num_objects = len(unique_ids)
        
        print(f"场景中的唯一物体ID: {unique_ids}")
        print(f"物体ID数量: {num_objects}")
        
        # 修复：确保unique_ids中的元素是标量
        # 如果unique_ids的元素是数组，取第一个元素
        if unique_ids.ndim > 1 or (unique_ids.size > 0 and isinstance(unique_ids[0], np.ndarray)):
            unique_ids = np.array([float(uid.item() if hasattr(uid, 'item') else uid[0] if hasattr(uid, '__len__') else uid) 
                                for uid in unique_ids])
            print(f"修复后的unique_ids: {unique_ids}")
        
        # 创建50+种颜色的组合色系
        def create_rich_colormap(n_colors):
            """创建丰富的颜色映射，支持50+种颜色"""
            
            # 基础颜色集合 - 20种基础色
            base_colors = [
                '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',  # 深蓝、橙、绿、红、紫
                '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf',  # 棕、粉、灰、橄榄、青
                '#aec7e8', '#ffbb78', '#98df8a', '#ff9896', '#c5b0d5',  # 浅蓝、浅橙、浅绿、浅红、浅紫
                '#c49c94', '#f7b6d3', '#c7c7c7', '#dbdb8d', '#9edae5'   # 浅棕、浅粉、浅灰、浅橄榄、浅青
            ]
            
            # 扩展颜色 - 增加更多变体
            extended_colors = [
                '#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7',  # 红橙、青绿、蓝、薄荷绿、浅黄
                '#DDA0DD', '#98D8C8', '#F7DC6F', '#BB8FCE', '#85C1E9',  # 梅花紫、薄荷、香蕉黄、丁香紫、天蓝
                '#F8C471', '#82E0AA', '#F1948A', '#85C1E9', '#D7BDE2',  # 桃、薄荷绿、珊瑚红、天蓝、淡紫
                '#A3E4D7', '#F9E79F', '#D5A6BD', '#AED6F1', '#A9DFBF',  # 水绿、浅黄、玫瑰、浅蓝、浅绿
                '#FAD7A0', '#E8DAEF', '#D6EAF8', '#FADBD8', '#D5F4E6',  # 杏、淡紫、淡蓝、粉、薄荷
                '#FCF3CF', '#EBDEF0', '#D4EDDA', '#F8D7DA', '#D1ECF1'   # 浅黄、浅紫、浅绿、浅红、浅蓝
            ]
            
            # 合并所有颜色
            all_colors = base_colors + extended_colors
            
            # 如果还需要更多颜色，使用HSV色彩空间生成
            if n_colors > len(all_colors):
                import colorsys
                hsv_colors = []
                for i in range(n_colors - len(all_colors)):
                    # 在HSV空间中均匀分布色相，保持高饱和度和亮度
                    hue = (i * 137.508) % 360 / 360.0  # 使用黄金角度避免相似颜色相邻
                    saturation = 0.7 + (i % 3) * 0.1   # 饱和度在0.7-0.9之间变化
                    value = 0.8 + (i % 2) * 0.2        # 亮度在0.8-1.0之间变化
                    rgb = colorsys.hsv_to_rgb(hue, saturation, value)
                    hsv_colors.append('#{:02x}{:02x}{:02x}'.format(
                        int(rgb[0]*255), int(rgb[1]*255), int(rgb[2]*255)))
                all_colors.extend(hsv_colors)
            
            return all_colors[:n_colors]
        
        # 生成颜色映射
        colors_list = create_rich_colormap(max(50, num_objects))
        
        # 为每个物体ID分配颜色
        id_to_color_idx = {uid: i % len(colors_list) for i, uid in enumerate(unique_ids)}
        
        # 创建颜色数组
        colors = []
        for obj_id in obj_ids:
            # 修复：确保obj_id是标量而不是数组
            scalar_id = None
            if hasattr(obj_id, 'item'):
                scalar_id = obj_id.item()  # 如果是numpy标量，转换为Python标量
            elif hasattr(obj_id, '__len__') and len(obj_id) > 0:
                scalar_id = obj_id[0]  # 如果是数组，取第一个元素
            else:
                scalar_id = float(obj_id)  # 直接转换为float
            color_idx = id_to_color_idx[scalar_id]
            color_hex = colors_list[color_idx]
            # 将十六进制颜色转换为RGB
            color_rgb = [int(color_hex[i:i+2], 16)/255.0 for i in (1, 3, 5)]
            colors.append(color_rgb)
        
        colors = np.array(colors)

        # 绘制点云
        scatter = ax.scatter(points[:, 0], points[:, 1], points[:, 2], 
                            c=colors, s=point_size, alpha=0.8)
        
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title(f'Point Cloud with {num_objects} Object IDs')

        # 坐标轴范围自适应
        max_range = np.array([points[:, 0].max()-points[:, 0].min(), 
                            points[:, 1].max()-points[:, 1].min(),
                            points[:, 2].max()-points[:, 2].min()]).max() / 2.0
        mid_x = (points[:, 0].max()+points[:, 0].min()) * 0.5
        mid_y = (points[:, 1].max()+points[:, 1].min()) * 0.5
        mid_z = (points[:, 2].max()+points[:, 2].min()) * 0.5
        ax.set_xlim(mid_x - max_range, mid_x + max_range)
        ax.set_ylim(mid_y - max_range, mid_y + max_range)
        ax.set_zlim(mid_z - max_range, mid_z + max_range)

        # 智能图例 - 分组显示
        if num_objects <= 15:
            # 少量物体：显示所有
            for i, uid in enumerate(unique_ids):
                color_hex = colors_list[i % len(colors_list)]
                color_rgb = [int(color_hex[j:j+2], 16)/255.0 for j in (1, 3, 5)]
                ax.scatter([], [], [], color=color_rgb, label=f'ID {int(uid)}', s=50)
            ax.legend(title='Object IDs', loc='upper right', fontsize=8, 
                     bbox_to_anchor=(1.15, 1))
        elif num_objects <= 30:
            # 中等数量：分两列显示
            legend_elements = []
            for i, uid in enumerate(unique_ids):
                color_hex = colors_list[i % len(colors_list)]
                color_rgb = [int(color_hex[j:j+2], 16)/255.0 for j in (1, 3, 5)]
                legend_elements.append(plt.Line2D([0], [0], marker='o', color='w', 
                                                markerfacecolor=color_rgb, markersize=6, 
                                                label=f'ID {int(uid)}'))
            ax.legend(handles=legend_elements, title=f'Objects ({num_objects})', 
                     loc='upper right', fontsize=6, ncol=2, 
                     bbox_to_anchor=(1.2, 1))
        else:
            # 大量物体：只显示前20个，并标注总数
            legend_elements = []
            for i, uid in enumerate(unique_ids[:20]):
                color_hex = colors_list[i % len(colors_list)]
                color_rgb = [int(color_hex[j:j+2], 16)/255.0 for j in (1, 3, 5)]
                legend_elements.append(plt.Line2D([0], [0], marker='o', color='w', 
                                                markerfacecolor=color_rgb, markersize=5, 
                                                label=f'ID {int(uid)}'))
            ax.legend(handles=legend_elements, 
                     title=f'Objects (showing 20/{num_objects})', 
                     loc='upper right', fontsize=5, ncol=3, 
                     bbox_to_anchor=(1.25, 1))

        # 调整布局以适应图例
        plt.tight_layout()

        # 保存或显示
        output_path = '/tmp/pointcloud_with_ids.png'
        if not is_main_thread or not has_display:
            plt.savefig(output_path, dpi=200, bbox_inches='tight', 
                       facecolor='white', edgecolor='none')
            print(f"点云可视化已保存到: {output_path}")
            plt.close(fig)
        else:
            plt.show()

        # 输出颜色统计信息
        print(f"\n颜色分配统计:")
        print(f"总物体数: {num_objects}")
        print(f"可用颜色数: {len(colors_list)}")
        print(f"颜色重复使用: {'是' if num_objects > len(colors_list) else '否'}")
    
    def _cal_test_score(self, points, normals, obj_ids, translations, rotations, visibility = False):
        """
        输入：
            points: 点云数据，形状为(N, 3)
            normals: 法向量数据，形状为(N, 3)
            obj_ids: 每个点对应的物体ID，形状为(N,)
            translations: 每个点对应的物体位姿平移向量，形状为(N, 3)，其值相当于该点对应的物体的中心坐标
            rotations: 每个点对应的物体位姿旋转矩阵，形状为(N, 9)，其值相当于该点对应的物体的旋转矩阵
        输出：
            scores: 各个点的评分，形状为(N,)
        """
        scores = np.zeros(points.shape[0])
        # 获取唯一的物体ID
        unique_ids = np.unique(obj_ids)
        
        # 存储分离后的点云
        separated_clouds = {}
        
        for obj_id in unique_ids:
            # 创建布尔掩码，选择属于当前物体的点
            mask = (obj_ids == obj_id)
            # 计算indices
            indices = np.where(mask)[0]
            
            # 提取对应的数据
            obj_data = {
                'points': points[mask],
                'normals': normals[mask],
                'obj_id': obj_id,
                'num_points': np.sum(mask),
                'origin_indices': indices,
                'translations': translations[mask],
                'rotations': rotations[mask]
            }
            
            separated_clouds[int(obj_id)] = obj_data
            
            # 计算每个物体上每个点的评分
            scores[mask] = self._cal_single_object_test_score(obj_data)
        return scores, separated_clouds
            
    def _cal_single_object_test_score(self, obj_data):
        points = obj_data['points'] # 形状：[N, 3]
        center = obj_data['translations'][0]  # 形状：[3]
        normals = obj_data['normals'] # 形状：[N, 3]
        # 计算每个点与中心点的向量（力矩）
        points_to_center = points - center # 形状：[N, 3]
        # 计算法向量与力矩的夹角
        dot_products = np.sum(normals * points_to_center, axis=1)  # 形状：[N]
        # 计算每个法向量的模长（向量长度）
        norm_magnitudes = np.linalg.norm(normals, axis=1) # 形状：[N]
        # 计算力矩的模长
        norm_points_to_center = np.linalg.norm(points_to_center, axis=1)  # 形状：[N]
        # 计算夹角的余弦值
        cos_angle = np.abs(dot_products / (norm_magnitudes * norm_points_to_center))
        # 防止数值误差导致的域错误
        return np.clip(cos_angle, 0.0, 1.0)
        
    
    def process_train_set(self, depth_img, segment_img, gt_file_path, output_file_path, individual_object_size_path):
        """
        处理单个训练样本，生成完整的H5数据集
        
        这是整个数据处理管道的核心方法，整合多模态输入数据，
        计算多维度吸取评分，最终生成标准化的训练数据。
        
        处理流程：
        1. 数据预处理：深度图转点云、分割信息提取
        2. 点云采样：FPS采样到目标点数
        3. 法向量估计：使用KDTree和半径搜索
        4. 多维评分计算：
           - 密封评分：基于预训练模型的KNN插值
           - 抗扭矩评分：考虑重力和力矩的物理建模
           - 碰撞评分：几何碰撞检测
           - 可见性评分：物体暴露面积比例
        5. 数据可视化：用于验证和调试
        6. H5格式保存：标准化输出
        
        参数:
            depth_img (numpy.ndarray): 深度图像，uint16格式
            segment_img (numpy.ndarray): 分割图像，EXR格式，包含物体ID
            gt_file_path (str): 真值CSV文件路径
            output_file_path (str): 输出H5文件路径
            individual_object_size_path (str): 物体尺寸标签CSV路径
            xyz_limit (list): 3D空间裁剪范围，可选
        """
        start_time = time.time()  # 性能监控起始时间
        
        # === 第1步：数据验证和预处理 ===
        # 验证深度图像格式和尺寸
        W = self.cam_info.intrinsic_matrix[0, 2] * 2  # 水平分辨率
        H = self.cam_info.intrinsic_matrix[1, 2] * 2
        expected_shape = (H, W)
        assert depth_img.shape == expected_shape and depth_img.dtype == np.uint16, \
            f"深度图像格式错误：期望{expected_shape}, uint16，实际{depth_img.shape}, {depth_img.dtype}"
        
        # 读取真值标签：物体位姿、ID、名称
        label_trans, label_rot, label_id, label_name = self._read_label_csv(gt_file_path)
        obj_num = label_trans.shape[0]  # 场景中物体数量
        
        # 计算前景掩码
        if obj_num == 1:
            step = 1.0  # 单个物体时，物体ID归一化步长为1
        else:    
            step = 1/(obj_num - 1)
        obj_ids = np.full(segment_img[:, :, 1].shape, 0, dtype=np.float32)
        valid_mask = (segment_img[:, :, 0] > 0.5)  # 前景点掩码
        
        # === 第2步：深度图转点云 ===
        # 提取非零深度像素的坐标（前景点）
        ys, xs = np.where(valid_mask)
        zs = depth_img[valid_mask]
        
        # 执行3D重建：像素坐标 + 深度 → 3D点云
        points = self._depth_to_pointcloud_optimized(xs, ys, zs, to_mm=False)
        
        # === 第3步：分割信息提取 ===
        # 从分割图像中提取每个点对应的物体ID
        # segment_img[:,:,2] == 1 表示前景点，segment_img[:,:,1] 包含归一化的物体ID
        segment_img_int = np.round(segment_img[:, :, 1] / step)
        # obj_ids 点所在物体在场景中的ID
        obj_ids = segment_img_int[valid_mask] # 每个像素对应的物体ID
        obj_ids = obj_ids.astype('int')  # 转换为整数类型

        # === 第4步：点云采样和标准化 ===
        num_pnt = points.shape[0]
        if num_pnt == 0:
            raise ValueError('没有前景点，跳过当前场景！')

        # normal_flip_mask代表滤波后的点云中有哪些点的法向量被翻转了，长度和滤波后的点云相同
        normalized_points, origin_world_normals, world_normals, filter_mask, normal_flip_mask= self._normalize_pointcloud(points, obj_ids, label_trans)
        
        # 根据滤波掩码更新物体ID数组 移除obj_ids中filter_mask不为True的点

        obj_ids = obj_ids[filter_mask]

        # 测试Debug
        # 计算suction_points的范围
        points_max = np.max(normalized_points, axis=0)  # 形状: (3,)
        points_min = np.min(normalized_points, axis=0)  # 形状: (3,)
        
        print(f"normalized_points各轴最大值: X={points_max[0]:.4f}, Y={points_max[1]:.4f}, Z={points_max[2]:.4f}")
        print(f"normalized_points各轴最小值: X={points_min[0]:.4f}, Y={points_min[1]:.4f}, Z={points_min[2]:.4f}")

        # 更新点数
        num_pnt = normalized_points.shape[0]
        if num_pnt == 0:
            raise ValueError('滤波后没有剩余点，跳过当前场景！')
            
        # 确保数组内存连续
        if not normalized_points.flags['C_CONTIGUOUS']:
            normalized_points = np.ascontiguousarray(normalized_points)
            world_normals = np.ascontiguousarray(world_normals)
            obj_ids = np.ascontiguousarray(obj_ids)
        
        # === 第5步：标签数据对齐 ===
        # 读取物体尺寸标签（可见面积比例）
        individual_object_size_lable = self.individual_label_csv(individual_object_size_path)[0]
        if individual_object_size_lable.size == 0:
            raise ValueError('尺寸标签文件为空！')
     
        # 情况1：点数过多，使用最远点采样(FPS)进行下采样
        if num_pnt > self.target_num_point:
            print(f"点数过多({num_pnt} > {self.target_num_point})，进行FPS降采样")
            
            # 转换为PyTorch张量并移到GPU（如果可用）
            points_transpose = torch.from_numpy(normalized_points.reshape(1, normalized_points.shape[0], normalized_points.shape[1])).float()
            points_transpose = points_transpose.cuda()
            
            # 执行最远点采样，保持点云的几何分布
            sampled_idx = furthest_point_sample(points_transpose, self.target_num_point).cpu().numpy().reshape(self.target_num_point)
            
            # 按采样索引获取数据
            normalized_points = normalized_points[sampled_idx]
            obj_ids = obj_ids[sampled_idx]
            world_normals = world_normals[sampled_idx]
            origin_world_normals = origin_world_normals[sampled_idx]
            normal_flip_mask = normal_flip_mask[sampled_idx]
            # 标签数据对齐
            try:
                points_label_trans = np.array([label_trans[obj_ids[i]] for i in range(len(obj_ids))])
                points_label_rot = np.array([label_rot[obj_ids[i]] for i in range(len(obj_ids))])
                points_label_id = np.array([label_id[obj_ids[i]] for i in range(len(obj_ids))])
            except (KeyError, IndexError) as e:
                print("label_trans: ", label_trans)
                print("label_rot: ", label_rot)
                raise ValueError(f"物体ID索引错误，请检查输入数据: {str(e)}")
            
            # 提取处理后的数据
            suction_points = normalized_points  # 吸取候选点
            suction_or = world_normals.astype(np.float32)  # 对应的法向量（已转换到世界坐标系）
            
            # 计算各种评分
            score_seal = self._cal_score_seal(suction_points, obj_ids, points_label_trans, points_label_rot, label_name)
            score_wrench = self._cal_score_wrench(suction_points, suction_or, points_label_trans, camera_info=self.cam_info)
            score_collision = self._cal_score_collision(suction_points, suction_or)
            score_visibility = self._cal_score_visibility(individual_object_size_lable, obj_ids)
            score_test, separated_clouds = self._cal_test_score(suction_points, suction_or, obj_ids, points_label_trans, points_label_rot)
            suction_or = origin_world_normals.astype(np.float32)  # 原始法向量（未翻转）
        # 情况2：点数不足，先计算分数再进行重复采样
        elif num_pnt < self.target_num_point:
            print(f"点数不足({num_pnt} < {self.target_num_point})，先计算分数再重采样")
            
            # 先对原始数据进行标签对齐
            try:
                points_label_trans = np.array([label_trans[obj_ids[i]] for i in range(len(obj_ids))])
                points_label_rot = np.array([label_rot[obj_ids[i]] for i in range(len(obj_ids))])
                points_label_id = np.array([label_id[obj_ids[i]] for i in range(len(obj_ids))])
            except (KeyError, IndexError) as e:
                print("label_trans: ", label_trans)
                print("label_rot: ", label_rot)
                raise ValueError(f"物体ID索引错误，请检查输入数据: {str(e)}")
            
            # 先计算原始点云的各种评分
            score_seal_original = self._cal_score_seal(normalized_points, obj_ids, points_label_trans, points_label_rot, label_name)
            score_wrench_original = self._cal_score_wrench(normalized_points, world_normals, points_label_trans, camera_info=self.cam_info)
            score_collision_original = self._cal_score_collision(normalized_points, world_normals)
            score_visibility_original = self._cal_score_visibility(individual_object_size_lable, obj_ids)
            score_test_original, separated_clouds = self._cal_test_score(normalized_points, world_normals, obj_ids, points_label_trans, points_label_rot)
            # 计算重复次数
            t = int(1.0 * self.target_num_point / num_pnt) + 1
            
            # 重复采样点云、法向量、物体ID和所有评分
            points_tile = np.tile(normalized_points, [t, 1])
            normalized_points = points_tile[:self.target_num_point]
            
            world_normals_tile = np.tile(world_normals, [t, 1])
            world_normals = world_normals_tile[:self.target_num_point]
            
            origin_world_normals_tile = np.tile(origin_world_normals, [t, 1])
            origin_world_normals = origin_world_normals_tile[:self.target_num_point]
            
            normal_flip_mask_tile = np.tile(normal_flip_mask, [t])
            normal_flip_mask = normal_flip_mask_tile[:self.target_num_point]
            
            obj_ids_tile = np.tile(obj_ids, [t])
            obj_ids = obj_ids_tile[:self.target_num_point]
            
            # 重复采样标签数据
            points_label_trans_tile = np.tile(points_label_trans, [t, 1])
            points_label_trans = points_label_trans_tile[:self.target_num_point]
            
            points_label_rot_tile = np.tile(points_label_rot, [t, 1])
            points_label_rot = points_label_rot_tile[:self.target_num_point]
            
            points_label_id_tile = np.tile(points_label_id, [t, 1])
            points_label_id = points_label_id_tile[:self.target_num_point]
            
            # 重复采样所有评分
            score_seal_tile = np.tile(score_seal_original, [t])
            score_seal = score_seal_tile[:self.target_num_point]
            
            score_wrench_tile = np.tile(score_wrench_original, [t])
            score_wrench = score_wrench_tile[:self.target_num_point]
            
            score_collision_tile = np.tile(score_collision_original, [t])
            score_collision = score_collision_tile[:self.target_num_point]
            
            score_visibility_tile = np.tile(score_visibility_original, [t])
            score_visibility = score_visibility_tile[:self.target_num_point]
            
            score_test_tile = np.tile(score_test_original, [t])
            score_test = score_test_tile[:self.target_num_point]
            
            # 提取处理后的数据
            suction_points = normalized_points
            suction_or = origin_world_normals.astype(np.float32)
            
        else:
            # 情况3：点数正好等于目标数量，无需采样
            print(f"点数正好等于目标数量({num_pnt} = {self.target_num_point})，无需采样")
            
            # 标签数据对齐
            try:
                points_label_trans = np.array([label_trans[obj_ids[i]] for i in range(len(obj_ids))])
                points_label_rot = np.array([label_rot[obj_ids[i]] for i in range(len(obj_ids))])
                points_label_id = np.array([label_id[obj_ids[i]] for i in range(len(obj_ids))])
            except (KeyError, IndexError) as e:
                print("label_trans: ", label_trans)
                print("label_rot: ", label_rot)
                raise ValueError(f"物体ID索引错误，请检查输入数据: {str(e)}")
            
            # 提取处理后的数据
            suction_points = normalized_points
            suction_or = world_normals.astype(np.float32)
            
            # 计算各种评分
            score_seal = self._cal_score_seal(suction_points, obj_ids, points_label_trans, points_label_rot, label_name)
            score_wrench = self._cal_score_wrench(suction_points, suction_or, points_label_trans, camera_info=self.cam_info)
            score_collision = self._cal_score_collision(suction_points, suction_or)
            score_visibility = self._cal_score_visibility(individual_object_size_lable, obj_ids)
            score_test, separated_clouds = self._cal_test_score(suction_points, suction_or, obj_ids, points_label_trans, points_label_rot)
            suction_or = origin_world_normals.astype(np.float32)
        # ------------------------------------------------------------------------------------step 5: save as h5 file
        # 保存所有点云、法线、分数等为h5格式
        # self._visualize_pointcloud(suction_points)
        normal_flip_mask.astype(np.float32)
        with h5py.File(output_file_path,'w') as f:
            f['points'] = suction_points  # 使用处理后的点云
            f['suction_or'] = suction_or  # 输入翻转前的法向量
            f['normal_flip_mask'] = normal_flip_mask  # 法向量翻转掩码, 需要训练
            f['suction_seal_scores'] = score_seal 
            f['suction_wrench_scores'] = score_wrench
            f['suction_feasibility_scores'] = score_collision
            f['individual_object_size_lable'] = score_visibility
            f['test_scores'] = score_test










