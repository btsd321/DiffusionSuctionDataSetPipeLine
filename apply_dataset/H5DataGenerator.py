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
import random        # 随机数生成

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
    
    def __init__(self, params_file_name, camera_info_file_name, objs_path, target_num_point=16384):
        """
        初始化数据生成器，加载相机参数和处理配置
        
        参数:
            params_file_name (str): 参数配置文件路径("parameter.json")
            target_num_point (int): 点云采样的目标点数，默认16384
                                   这个数量平衡了计算效率和数据质量
        """
        # 加载深度范围等关键参数
        self.params = self._load_parameters(params_file_name)
        # 加载相机内参
        self.cam_info = CameraInfo(camera_info_file_name)
        # 设置点云采样目标数量，确保数据一致性
        self.target_num_point = target_num_point
        self.objs_path = objs_path

    def _depth_to_pointcloud_optimized(self, us, vs, zs, to_mm=False, xyz_limit=None):
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
        Xcs = np.reshape(Xcs, (-1, 1))
        Ycs = np.reshape(Ycs, (-1, 1))
        Zcs = np.reshape(Zcs, (-1, 1))
        points = np.concatenate([Xcs, Ycs, Zcs], axis=-1)
        
        # 可选：根据xyz范围裁剪点云，去除工作空间外的无关点
        if xyz_limit is not None:
            # X轴裁剪
            if xyz_limit[0] is not None:
                xmin, xmax = xyz_limit[0]
                if xmin is not None:
                    idx = np.where(points[:, 0] > xmin)
                    points = points[idx]
                if xmax is not None:
                    idx = np.where(points[:, 0] < xmax)
                    points = points[idx]
            # Y轴裁剪
            if xyz_limit[1] is not None:
                ymin, ymax = xyz_limit[1]
                if ymin is not None:
                    idx = np.where(points[:, 1] > ymin)
                    points = points[idx]
                if ymax is not None:
                    idx = np.where(points[:, 1] < ymax)
                    points = points[idx]
            # Z轴裁剪
            if xyz_limit[2] is not None:
                zmin, zmax = xyz_limit[2]
                if zmin is not None:
                    idx = np.where(points[:, 2] > zmin)
                    points = points[idx]
                if zmax is not None:
                    idx = np.where(points[:, 2] < zmax)
                    points = points[idx]
                    
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
        # 坐标归一化到物体坐标系
        suction_points_normalization = np.matmul((suction_points - label_trans).reshape(self.target_num_point,1,3), label_rot.reshape(-1,3,3) )
        suction_points_normalization = suction_points_normalization.reshape(self.target_num_point, 3)
        suction_seal_scores  = np.zeros((self.target_num_point, ))
        # 计算吸取点的密封分数
        for index in range(len(label_name)):
            # 读取每个物体的稀疏点和分数
            annotation = np.load(os.path.join(self.objs_path, label_name[index], "labels.npz"))
            object_sparse_point = annotation['points']
            anno_points = annotation['points']
            anno_scores = annotation['scores']

            suction_points_normalization_id = suction_points_normalization[obj_ids == index]
            if suction_points_normalization_id.shape[0] == 0:
                continue
            suction_points_normalization_id_knn = torch.from_numpy(suction_points_normalization_id).float()
            suction_points_normalization_id_knn = suction_points_normalization_id_knn.cuda()
            anno_points_knn = torch.from_numpy(anno_points).float()
            anno_points_knn = anno_points_knn.cuda()
            # knn最近邻查找, 获取吸取点的密封分数
            indices, dist=knn(anno_points_knn, suction_points_normalization_id_knn, k=1)
            dist=dist.cpu().numpy().reshape(dist.shape[-1])
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
        import matplotlib.pyplot as plt
        plt.hist(score_seal, bins=100)
        plt.title("Histogram of Data")
        plt.xlabel("Value")
        plt.ylabel("Frequency")
        plt.show()
        return
    
    def _cal_score_wrench(self, suction_points, suction_or, label_trans):
        '''
        计算抗扭矩评分，抗扭矩评分用于判断吸盘在特定姿势下是否无法抵抗重力
        '''
        # 计算吸取点的抗扭分数(考虑重力和吸盘姿态)
        k = 30
        radius = 0.01
        wrench_thre = k * radius * np.pi * np.sqrt(2)
        suction_wrench_scores = []
        for index_temp,suction_points_temp in enumerate(suction_points):
            label_trans_temp = label_trans[index_temp]
            suction_or_temp = suction_or[index_temp]
            center = label_trans_temp
            gravity = np.array([[0, 0, 1]], dtype=np.float32) * 9.8  # 重力方向
            suction_axis = viewpoint_to_matrix_z(suction_or_temp)  # (3, 3)
            suction2center = (center - suction_points_temp)[np.newaxis, :]
            coord = np.matmul(suction2center, suction_axis)
            gravity_proj = np.matmul(gravity, suction_axis)
            torque_y = gravity_proj[0, 0] * coord[0, 2] - gravity_proj[0, 2] * coord[0, 0]
            torque_x = -gravity_proj[0, 1] * coord[0, 2] + gravity_proj[0, 2] * coord[0, 1]
            torque = np.sqrt(torque_x**2 + torque_y**2)
            score = 1 - min(1, torque / wrench_thre)
            suction_wrench_scores.append(score)
        suction_wrench_scores = np.array(suction_wrench_scores)
        return suction_wrench_scores
    
    def _cal_score_collision(self, suction_points, suction_or):
        '''
        计算碰撞评分，碰撞评分用于评估吸取点与其他物体的几何碰撞情况
        '''
        # 计算吸取点的可行性分数(碰撞检测)
        height = 0.1
        radius = 0.01
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
            mask2 = ((target[:,0] > 0.005) & (target[:,0] < height))
            mask = np.any(mask1 & mask2)
            suction_feasibility_scores.append(mask)
        suction_feasibility_scores = ~np.array(suction_feasibility_scores)
        return suction_feasibility_scores
    
    def _cal_score_visibility(self, label):
        '''
        计算可见性评分，可见性评分用于定量反映场景中对象的被遮挡程度
        '''
        return label
    
    def _cal_suction_score(self):
        '''
        计算吸取分数，公式：总分数=Seal * Wrench * Collision * Visibility(密封评分*抗扭矩评分*碰撞评分*可见性评分)
        '''
        score_seal = self._cal_score_seal()
        score_wrench = self._cal_score_wrench()
        score_collision = self._cal_score_collision()
        score_visibility = self._cal_score_visibility()
        score = score_seal * score_wrench * score_collision * score_visibility
        return score

    def process_train_set(self, depth_img, segment_img, gt_file_path, output_file_path, individual_object_size_path, xyz_limit=None):
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
        
        # === 第2步：深度图转点云 ===
        # 提取非零深度像素的坐标（前景点）
        xs = np.where(depth_img != 0)[1]  # u坐标（水平）
        ys = np.where(depth_img != 0)[0]  # v坐标（垂直）
        zs = depth_img[depth_img != 0]    # 深度值
        
        # 执行3D重建：像素坐标 + 深度 → 3D点云
        points = self._depth_to_pointcloud_optimized(xs, ys, zs, to_mm=False, xyz_limit=xyz_limit)
        
        # === 第3步：分割信息提取 ===
        # 从分割图像中提取每个点对应的物体ID
        # segment_img[:,:,2] == 1 表示前景点，segment_img[:,:,1] 包含归一化的物体ID
        obj_ids = np.round(segment_img[:, :, 1][segment_img[:, :, 2] == 1] * (obj_num - 1)).astype('int')
        
        # === 第4步：点云采样和标准化 ===
        num_pnt = points.shape[0]
        if num_pnt == 0:
            print('警告：没有前景点，跳过当前场景！')
            return
            
        # 情况1：点数不足，通过重复采样达到目标数量
        if num_pnt <= self.target_num_point:
            t = int(1.0 * self.target_num_point / num_pnt) + 1
            points_tile = np.tile(points, [t, 1])  # 重复点云
            points = points_tile[:self.target_num_point]
            obj_ids_tile = np.tile(obj_ids, [t])  # 重复物体ID
            obj_ids = obj_ids_tile[:self.target_num_point]
            
        # 情况2：点数过多，使用最远点采样(FPS)进行下采样
        if num_pnt > self.target_num_point:
            # 转换为PyTorch张量并移到GPU（如果可用）
            points_transpose = torch.from_numpy(points.reshape(1, points.shape[0], points.shape[1])).float()
            points_transpose = points_transpose.cuda()
            
            # 执行最远点采样，保持点云的几何分布
            sampled_idx = furthest_point_sample(points_transpose, self.target_num_point).cpu().numpy().reshape(self.target_num_point)
            points = points[sampled_idx]
            obj_ids = obj_ids[sampled_idx]

        # === 第5步：标签数据对齐 ===
        # 读取物体尺寸标签（可见面积比例）
        individual_object_size_lable = self.individual_label_csv(individual_object_size_path)[0]
        
        # 根据采样后的点云，重新对齐所有标签数据
        # 确保每个点都有对应的物体位姿、ID和尺寸信息
        label_trans = label_trans[obj_ids]  # 物体位置对应到每个点
        label_rot = label_rot[obj_ids]      # 物体旋转对应到每个点
        label_id = label_id[obj_ids]        # 物体ID对应到每个点
        individual_object_size_lable = individual_object_size_lable[obj_ids]  # 尺寸标签对应到每个点

        # === 第6步：法向量估计 ===
        # 构建Open3D点云对象用于法向量计算
        pc_o3d = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))
        
        # 使用半径搜索估计每个点的表面法向量
        # 半径0.015米是经验值，平衡计算精度和效率
        pc_o3d.estimate_normals(
            o3d.geometry.KDTreeSearchParamRadius(0.015), 
            fast_normal_computation=False  # 使用精确计算保证质量
        )
        
        # 统一法向量方向：都指向负Z轴方向（向下）
        # 这对吸取任务很重要，因为吸盘通常从上往下接近物体
        pc_o3d.orient_normals_to_align_with_direction(np.array([0., 0., -1.]))
        pc_o3d.normalize_normals()  # 标准化为单位向量
        
        # 提取处理后的数据
        suction_points = points  # 吸取候选点
        suction_or = np.array(pc_o3d.normals).astype(np.float32)  # 对应的法向量

        # 法线可视化
        show_point_temp=o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))
        show_point_temp.colors = o3d.utility.Vector3dVector([[0, 0, 1]  for i in range(points.shape[0])])
        vis_list = [  show_point_temp   ]
        # 可视化前100个吸取点的法线方向
        for idx in range(len(suction_points[0:100])):
            suction_point = suction_points[idx]
            anno_normal = suction_or[idx]
            suction_score = individual_object_size_lable[idx]
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
        # 绘制尺寸标签直方图
        import matplotlib.pyplot as plt
        individual_object_size_lable_temp = individual_object_size_lable*100
        plt.hist(individual_object_size_lable_temp.astype(int), bins=100)
        plt.title("Histogram of Data")
        plt.xlabel("Value")
        plt.ylabel("Frequency")
        plt.show()

        # 密封评分
        score_seal = self._cal_score_seal(suction_points, obj_ids, label_trans, label_rot, label_name)
        self._score_seel_visiualization(score_seal, suction_points, suction_or)

        # 抗扭矩评分
        score_wrench = self._cal_score_wrench(suction_points, suction_or,  label_trans)

        # 碰撞评分
        score_collision = self._cal_score_collision(suction_points, suction_or)

        # 可见性评分
        score_visibility = self._cal_score_visibility(individual_object_size_lable)

        # 综合所有分数, 得到最终分数并排序
        score_all = score_seal * score_wrench * score_collision * score_visibility

        sorted_indices = np.argsort(score_all)[::-1]
        score_all_asort = score_all[sorted_indices]
        points_asort = points[sorted_indices]
        suction_points_asort = suction_points[sorted_indices]
        suction_or_asort = suction_or[sorted_indices]

        # 可视化最终排序后的吸取点
        show_point_temp=o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points_asort))
        colors_temp = [[0, 0, 1]  for i in range(points.shape[0])]
        show_point_temp.colors = o3d.utility.Vector3dVector(colors_temp)
        vis_list = [  show_point_temp   ]
        for idx in range(len(suction_points_asort[0:600])):
            suction_point = suction_points_asort[idx]
            anno_normal = suction_or_asort[idx]
            suction_score = score_all_asort[idx]
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
        o3d.visualization.draw_geometries(vis_list, width=800,   height=600)
        # 绘制最终分数直方图
        import matplotlib.pyplot as plt
        plt.hist(score_all, bins=100)
        plt.title("Histogram of Data")
        plt.xlabel("Value")
        plt.ylabel("Frequency")
        plt.show()

        # ------------------------------------------------------------------------------------step 5: save as h5 file
        # 保存所有点云、法线、分数等为h5格式
        with h5py.File(output_file_path,'w') as f:
            f['points'] = points
            f['suction_or'] = suction_or
            f['suction_seal_scores'] = score_seal
            f['suction_wrench_scores'] = score_wrench
            f['suction_feasibility_scores'] = score_collision
            f['individual_object_size_lable'] = score_visibility










