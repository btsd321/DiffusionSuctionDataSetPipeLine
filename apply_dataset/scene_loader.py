# 包裹场景加载器

import os 
import sys
import numpy as np
import common_info
import csv
import cv2
import Imath
import OpenEXR
import open3d as o3d
import torch
from pointnet2_ops_lib.pointnet2_ops.pointnet2_utils import furthest_point_sample

import package # 自定义的包裹类



class SceneLoader:
    def __init__(self, depth_image_path: str, segment_image_path: str, gt_file_path: str, individual_object_size_path: str, \
                 common_info: common_info.CommonInfo):
        self._depth_image_path = depth_image_path
        self._segment_image_path = segment_image_path
        self._gt_file_path = gt_file_path
        self._individual_file_path = individual_object_size_path
        self._camera_info = common_info.get_camera_info()
        self._parameters = common_info.get_parameters()

        # 参数初始化
        self._depth_image = None
        self._segment_image = None
        self._obj_num = 0
        self._orgin_obj_ids = None  # 原始点云中每个点对应的物体在场景中的ID
        self._obj_ids = None # 滤波后点云中每个点对应的物体在场景中的ID
        self._label_trans = None
        self._label_rot = None
        self._label_id = None
        self._label_name = None
        self._label_visibility = None # 物体可见性标签
        self._origin_points = None # 原始点云数据
        self._points_in_camera = None # 滤波后的点云数据
        self._normals_in_camera = None # 滤波后的点云法向量
        self._points = None # 滤波后点云在世界坐标系中的坐标
        self._normals_before_flip = None # 滤波后点云在世界坐标系中的法向量（未翻转）
        self._normals = None # 滤波后点云在世界坐标系中的法向量（已翻转）
        self._opposite_direction_mask = None # 反向法向量掩码(相对于滤波后的法向量)
        self._visibilitys = None # 场景中物体可见性，形状(N,)
        self._packages = {} # 存储场景中的所有包裹

        # 开始解析
        self._read_gt_label_csv()
        self._read_individual_label_csv()
        self._generate_points_cloud()
        self._segment_package()

    def _segment_package(self):
        '''
        分割包裹点云
        '''
        self._normals = np.zeros_like(self._normals_before_flip, dtype=np.float32) # 初始化法向量为零向量
        unique_ids = np.unique(self._obj_ids)
        self._opposite_direction_mask = np.zeros(self._points.shape[0], dtype=bool)
        self._visibilitys = np.zeros(self._points.shape[0], dtype=np.float32)
        for obj_id in unique_ids:
            # 创建布尔掩码，选择属于当前物体的点
            mask = (self._obj_ids == obj_id)
            # 计算indices
            indices = np.where(mask)[0]
            if len(indices) == 0:
                continue

            package_init_data = {
                "points": self._points[indices],
                "normals_before_flip": self._normals_before_flip[indices],
                "center": self._label_trans[obj_id],
                "rotation": self._label_rot[obj_id],
                "name": self._label_name[obj_id],
                "visibility": self._label_visibility[0, obj_id],
                "mask": mask,
                "indices": indices
            }

            self._packages[obj_id] = package.Package(package_init_data)
            # 针对每个包裹翻转法向量使得法向量指向包裹中心
            self._normals[mask], self._opposite_direction_mask[mask] = self._packages[obj_id].flip_normals()
            # 获取包裹的可见性数据，确保形状匹配
            pkg_visibility = self._packages[obj_id].get_visibility()
            self._visibilitys[mask] = np.tile(pkg_visibility, np.sum(mask))

    def _read_individual_label_csv(self):
        """
        读取单个物体的尺寸标签CSV文件
        
        该文件包含每个物体的可见面积比例信息，用于评估
        吸取任务中物体的暴露程度和可操作性。
        
        参数:
            file_name (str): 物体尺寸标签CSV文件路径
            
        返回:
            numpy.ndarray: 物体尺寸标签数组，数据类型为float32
        """
        with open(self._individual_file_path, 'r') as csv_file:  
            all_lines = csv.reader(csv_file) 
            list_file = [i for i in all_lines]  
        # 直接转换为float32数组，不排除标题行（因为数据文件格式）
        self._label_visibility = np.array(list_file).astype('float32')
        return self._label_visibility

    def _read_gt_label_csv(self):
        """
        读取场景真值标签CSV文件，获取物体位姿信息
        
        解析CSV文件中的物体6D位姿数据（位置+旋转矩阵），
        这些数据来自物理仿真或标注工具。
        
        参数:
            file_name (str): 真值CSV文件路径
            
        返回:
            tuple: (平移向量, 旋转矩阵, 物体ID, 物体名称)
                - label_trans: (M, 3) 物体位置坐标, M为场景中物体数量
                - label_rot: (M, 9) 旋转矩阵（展平为9维）, M为场景中物体数量
                - label_id: (M, 1) 物体唯一标识符, M为场景中物体数量
                - label_name: (M,) 物体名称列表, M为场景中物体数量
        """
        with open(self._gt_file_path, 'r') as csv_file:  
            all_lines = csv.reader(csv_file) 
            list_file = [i for i in all_lines]  
        
        # 排除CSV表头，获取数据行
        array_file = np.array(list_file)[1:]
        num_obj = int(array_file.shape[0])
        
        # 解析各列数据：位置(3列) + 旋转矩阵(9列) + ID(1列) + 名称(1列)
        self._label_trans = array_file[:, 2:5].astype('float32')    # 物体xyz位置
        self._label_rot = array_file[:, 5:14].astype('float32')     # 3x3旋转矩阵展平
        self._label_id = array_file[:, 1:2].astype('float32')       # 物体编号
        self._label_name = array_file[:, 0]                         # 物体名称
        self._obj_num = self._label_name.shape[0]  # 场景中物体数量
    
    def _read_exr_to_numpy(self):
        exr_file = OpenEXR.InputFile(self._segment_image_path)
        header = exr_file.header()
        dw = header['dataWindow']
        width = dw.max.x - dw.min.x + 1
        height = dw.max.y - dw.min.y + 1
        channels = ['R', 'G', 'B']
        pt = Imath.PixelType(Imath.PixelType.FLOAT)
        data = [np.frombuffer(exr_file.channel(c, pt), dtype=np.float32) for c in channels]
        img = np.stack([d.reshape(height, width) for d in data], axis=-1)
        return img
    
    def _generate_points_cloud(self):
        # 读取深度图
        self._depth_image = cv2.imread(self._depth_image_path, cv2.IMREAD_UNCHANGED)
        if self._depth_image is None:
            raise ValueError(f"无法读取深度图像文件: {self._depth_image_path}")
        # 读取分割图
        self._segment_image = self._read_exr_to_numpy()

        # 计算前景掩码
        step = None
        if self._obj_num == 1:
            step = 1.0  # 单个物体时，物体ID归一化步长为1
        else:    
            step = 1/(self._obj_num - 1)
        obj_ids = np.full(self._segment_image[:, :, 1].shape, 0, dtype=np.float32)
        valid_mask = (self._segment_image[:, :, 0] > 0.5)  # 前景点掩码
        
        # 提取非零深度像素的坐标（前景点）
        ys, xs = np.where(valid_mask)
        zs = self._depth_image[valid_mask]
        self._origin_points = self._depth_to_pointcloud_optimized(xs, ys, zs, to_mm=False)

        segment_img_int = np.round(self._segment_image[:, :, 1] / step)
        # obj_ids 点所在物体在场景中的ID
        orgin_obj_ids = segment_img_int[valid_mask] # 每个像素对应的物体ID
        self._orgin_obj_ids = orgin_obj_ids.astype('int')  # 转换为整数类型

        self._points_in_camera, self._normals_in_camera, inlier_mask = self._filter_point_cloud()
        self._obj_ids = self._orgin_obj_ids[inlier_mask]  # 滤波后点云中每个点对应的物体ID

        # 转换到世界坐标系
        self._points, self._normals_before_flip = self._transform_to_world_coordinates()

    def _transform_to_world_coordinates(self):
        """
        将点云从相机坐标系转换到世界坐标系
        
        使用物体的位姿信息（平移和旋转）将点云坐标转换为世界坐标。
        这是数据处理管道的关键步骤之一。
        
        返回:
            numpy.ndarray: 转换后的点云坐标，形状为(N, 3)
        """
        # 将3D点扩展为齐次坐标（添加第4维度为1）
        ones = np.ones((self._points_in_camera.shape[0], 1))
        points_homo = np.hstack([self._points_in_camera, ones])  # 形状: (N, 4)
        
        #外参矩阵为W2C矩阵即世界坐标系转相机坐标系的矩阵，因此需要先计算C2W
        c2w = np.linalg.inv(self._camera_info.extrinsic_matrix)
        
        # 使用4x4外参矩阵进行变换
        # 外参矩阵将相机坐标系转换为世界坐标系
        points_world_homo = c2w @ points_homo.T # 形状: (4, N)
        
        # 提取前3个维度，去除齐次坐标
        points_world = (points_world_homo.T)[:, :3] 

        # 法向量只需要旋转变换，提取旋转矩阵部分
        rotation_matrix = self._camera_info.extrinsic_matrix[:3, :3].T
        
        # 应用旋转变换
        normals_world = (rotation_matrix @ self._normals_in_camera.T).T # 形状: (N, 3)
        
        # 重新归一化法向量
        norms = np.linalg.norm(normals_world, axis=1, keepdims=True)
        if np.any(norms == 0):  # 避免除零
            raise ValueError("法向量归一化时存在零向量")
        normals_world = normals_world / norms  
        
        return points_world, normals_world
        

    def _filter_point_cloud(self, nb_points = 16, filter_radius = 0.01, z_threshold = 0.7):
        try:
            # 创建Open3D点云对象
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(self._origin_points)
            
            # 应用半径滤波
            pcd_filtered, inlier_indices_radius = pcd.remove_radius_outlier(
                nb_points, filter_radius
            )
            
            # 转换回numpy数组
            radius_filtered_points = np.asarray(pcd_filtered.points)
            
            # 创建正确的索引掩码
            inlier_mask1 = np.zeros(self._origin_points.shape[0], dtype=bool)
            inlier_mask1[inlier_indices_radius] = True
            
            # print(f"半径滤波: 原始点数 {self._origin_points.shape[0]}, 过滤后点数 {radius_filtered_points.shape[0]}")
            # print(f"掩码统计: True={np.sum(inlier_mask1)}, False={np.sum(~inlier_mask1)}")
            # self._obj_ids_after_filter_1 = self._orgin_obj_ids[inlier_mask1]  # 滤波后点云中每个点对应的物体ID

            # 计算法向量（只对过滤后的点）
            radius_filter_normals = None
            if radius_filtered_points.shape[0] > 0:
                pc_o3d = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(radius_filtered_points))
                pc_o3d.estimate_normals(
                    o3d.geometry.KDTreeSearchParamRadius(0.015), 
                    fast_normal_computation=False
                )
                
                pc_o3d.normalize_normals()
                
                radius_filter_normals = np.array(pc_o3d.normals).astype(np.float32)
            else:
                radius_filter_normals = np.array([]).reshape(0, 3)

            # 直通滤波
            # 创建Z轴过滤掩码：保留Z坐标小于等于阈值的点
            z_filter_mask = radius_filtered_points[:, 2] <= z_threshold
            
            # 过滤点云和法向量
            z_filtered_points = radius_filtered_points[z_filter_mask]
            z_filtered_normals = radius_filter_normals[z_filter_mask]
            
            # 更新累积过滤掩码
            # 创建新的累积掩码，标记哪些原始点被保留
            filter_mask = np.zeros_like(inlier_mask1, dtype=bool)
            
            # 在之前被保留的点中，进一步标记通过Z轴过滤的点
            z_passed_positions = np.where(z_filter_mask)[0]
            z_passed_indices = np.array(inlier_indices_radius)[z_passed_positions]
            z_passed_indices = z_passed_indices.astype(int)
            filter_mask[z_passed_indices] = True
            
            return z_filtered_points, z_filtered_normals, filter_mask
        except Exception as e:
            print(f"半径滤波失败，使用原始点云: {e}")
            identity_mask = np.ones(self._origin_points.shape[0], dtype=bool)
            dummy_normals = np.zeros((self._origin_points.shape[0], 3), dtype=np.float32)
            dummy_normals[:, 2] = -1
            return self._origin_points, dummy_normals, identity_mask
        
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
        fx = self._camera_info.intrinsic_matrix[0, 0]
        fy = self._camera_info.intrinsic_matrix[1, 1]
        cx = self._camera_info.intrinsic_matrix[0, 2]  # x方向主点坐标
        cy = self._camera_info.intrinsic_matrix[1, 2]  # y方向主点坐标
        clip_start = self._parameters['clip_start']  # 近裁剪面距离
        clip_end = self._parameters['clip_end']      # 远裁剪面距离
        
        # 将归一化深度值转换为真实距离（米）
        # 深度图中的值通常是归一化的，需要映射到真实距离范围
        Zline = clip_start + (zs/self._parameters['max_val_in_depth']) * (clip_end - clip_start)
        
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

    def downsample(self, output_points_num = 16384):
        '''
        场景下采样时同时对包裹进行下采样
        '''
        if self._points.shape[0] <= output_points_num:
            return
        # 转换为PyTorch张量并移到GPU（如果可用）
        if not self._points.flags['C_CONTIGUOUS']:
            self._points = np.ascontiguousarray(self._points)
        points_transpose = torch.from_numpy(self._points.reshape(1, self._points.shape[0], self._points.shape[1])).float()
        points_transpose = points_transpose.cuda()
        
        # 执行最远点采样，保持点云的几何分布
        sampled_idx = furthest_point_sample(points_transpose, output_points_num).cpu().numpy().reshape(output_points_num)
        self._points = self._points[sampled_idx]
        self._normals = self._normals[sampled_idx]
        self._obj_ids = self._obj_ids[sampled_idx]
        self._normals_before_flip = self._normals_before_flip[sampled_idx]
        self._opposite_direction_mask = self._opposite_direction_mask[sampled_idx]
        self._visibilitys = self._visibilitys[sampled_idx]
        for obj_id, pkg in self._packages.items():
            # 获取包裹在场景中的掩码
            original_mask = pkg.get_mask_in_scene()
            
            # 向量化操作：创建新的掩码
            # 检查sampled_idx中的每个索引是否在original_mask范围内且为True
            valid_indices_mask = sampled_idx < len(original_mask)  # 防止索引越界
            new_mask = np.zeros(len(sampled_idx), dtype=bool)
            
            # 只对有效索引进行检查
            if np.any(valid_indices_mask):
                valid_sampled_idx = sampled_idx[valid_indices_mask]
                new_mask[valid_indices_mask] = original_mask[valid_sampled_idx]
            
            # 更新包裹的数据
            pkg._Package__mask_in_scene = new_mask
            pkg._Package__indices_in_scene = np.where(new_mask)[0]
            # 根据新掩码提取包裹的点云数据
            if np.any(new_mask):
                pkg._Package__points = self._points[new_mask]
                pkg._Package__normals = self._normals[new_mask]
                pkg._Package__normals_before_flip = self._normals_before_flip[new_mask]
                pkg._Package__opposite_direction_mask = self._opposite_direction_mask[new_mask]
                pkg._Package__visibility = self._visibilitys[new_mask]
            else:
                # 如果包裹没有点被保留，创建空数组
                pkg._Package__points = np.array([], dtype=np.float32).reshape(0, 3)
                pkg._Package__normals = np.array([], dtype=np.float32).reshape(0, 3)
                pkg._Package__normals_before_flip = np.array([], dtype=np.float32).reshape(0, 3)
                pkg._Package__opposite_direction_mask = np.array([], dtype=bool)
                pkg._Package__visibility = np.array([], dtype=np.float32)

    def get_pointcloud(self):
        # 获取点云数据
        return self._points
    
    def get_normals(self):
        # 获取点云法向量
        return self._normals

    def get_packages(self) -> dict:
        # 获取场景中的所有包裹
        return self._packages
    
    def get_normals_before_flip(self):
        # 获取翻转前的法向量
        return self._normals_before_flip
    
    def get_opposite_direction_mask(self):
        # 获取翻转前的法向量
        return self._opposite_direction_mask

    def get_visibility(self):
        # 获取场景中每个点的可见性，返回形状为(N, 3)
        return self._visibilitys