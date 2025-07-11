import os
import sys
FILE_PATH = os.path.abspath(__file__)
FILE_DIR_generate_dataset = os.path.dirname(FILE_PATH)
FILE_DIR = os.path.dirname(FILE_DIR_generate_dataset)
OBJ_PATH =  os.path.join(FILE_DIR, 'OBJ')

import json
import math
import numpy as np
import cv2
import os
import time
import torch
import open3d as o3d
from torch_cluster import knn

# from tf_ops.sampling.tf_sampling import gather_point, farthest_point_sample
from pointnet2_ops_lib.pointnet2_ops.pointnet2_utils import furthest_point_sample
import open3d as o3d
import h5py
import time
import csv
import random

def viewpoint_to_matrix_x(towards):
    # 根据朝向向量生成以x轴为主的旋转矩阵
    axis_x = towards
    axis_y = np.array([-axis_x[1], axis_x[0], 0])
    axis_x = axis_x / np.linalg.norm(axis_x)
    axis_y = axis_y / np.linalg.norm(axis_y)
    axis_z = np.cross(axis_x, axis_y)
    R2 = np.c_[axis_x, np.c_[axis_y, axis_z]]
    matrix = R2
    return matrix

def viewpoint_to_matrix_z(towards):
    # 根据朝向向量生成以z轴为主的旋转矩阵
    n = towards
    new_z = n
    new_y = np.array((new_z[1], -new_z[0], 0), dtype=np.float64)
    new_y = new_y / np.linalg.norm(new_y)
    new_z = new_z / np.linalg.norm(new_z)
    new_x = np.cross(new_y, new_z)
    new_x = new_x / np.linalg.norm(new_x)
    new_x = np.expand_dims(new_x, axis=1)
    new_y = np.expand_dims(new_y, axis=1)
    new_z = np.expand_dims(new_z, axis=1)  
    rot_matrix = np.concatenate((new_x, new_y, new_z), axis=-1)
    return rot_matrix

class H5DataGenerator(object):
    def __init__(self, params_file_name, target_num_point = 16384):
        '''
        初始化数据生成器, 加载相机参数
        输入参数:
            params_file_name: 参数文件路径("parameter.json")
            target_num_point: 采样点的目标数量, 默认16384
        '''
        self.params = self._load_parameters(params_file_name)
        self.target_num_point = target_num_point

    def _depth_to_pointcloud_optimized(self, us, vs, zs, to_mm = False, xyz_limit=None):
        '''
        深度图像像素坐标转点云
        输入参数:
            us: u坐标的np数组
            vs: v坐标的np数组
            zs: z坐标的np数组
            to_mm: 若为True则*1000.0, 单位转为毫米
            xyz_limit: 若为None则xyz无限制。典型格式为[ [xmin, xmax], [ymin, ymax], [zmin, zmax] ]
        输出:
            x, y, z
        '''
        assert len(us) == len(vs) == len(zs)
        camera_info = self.params
        fx = camera_info['fu']
        fy = camera_info['fv']
        cx = camera_info['cu']
        cy = camera_info['cv']
        clip_start = camera_info['clip_start']
        clip_end = camera_info['clip_end']
        # 将深度值归一化到真实距离
        Zline = clip_start + (zs/camera_info['max_val_in_depth']) * (clip_end - clip_start)
        Zcs = Zline/np.sqrt(1+ np.power((us-cx)/fx,2) + np.power((vs-cy)/fy,2))
        if to_mm:
            Zcs *= 1000
        Xcs = (us - cx) * Zcs / fx
        Ycs = (vs - cy) * Zcs / fy
        Xcs = np.reshape(Xcs, (-1, 1))
        Ycs = np.reshape(Ycs, (-1, 1))
        Zcs = np.reshape(Zcs, (-1, 1))
        points = np.concatenate([Xcs, Ycs, Zcs], axis=-1)
        # 可选：根据xyz范围裁剪点云
        if xyz_limit is not None:
            if xyz_limit[0] is not None:
                xmin, xmax = xyz_limit[0]
                if xmin is not None:
                    idx = np.where( points[:, 0]>xmin )
                    points = points[idx]
                if xmax is not None:
                    idx = np.where( points[:, 0]<xmax )
                    points = points[idx]
            if xyz_limit[1] is not None:
                ymin, ymax = xyz_limit[1]
                if ymin is not None:
                    idx = np.where( points[:, 1]>ymin )
                    points = points[idx]
                if ymax is not None:
                    idx = np.where( points[:, 1]<ymax )
                    points = points[idx]
            if xyz_limit[2] is not None:
                zmin, zmax = xyz_limit[2]
                if zmin is not None:
                    idx = np.where( points[:, 2]>zmin )
                    points = points[idx]
                if zmax is not None:
                    idx = np.where( points[:, 2]<zmax )
                    points = points[idx]
        return points

    def _load_parameters(self, params_file_name):
        '''
        加载相机参数文件
        输入参数:
            params_file_name: 参数文件路径("parameter.json")
        返回:
            params: 参数字典
        '''
        params = {}
        with open(params_file_name,'r') as f:
            config = json.load(f)
            params = config
        return params 

    def _read_label_csv(self, file_name):
        '''
        读取场景标签csv, 返回平移、旋转、id、名称
        输入参数:
            file_name: 真值csv文件路径
        输出:
            label_trans: numpy数组, 形状为(num_obj+1)*3, 0号位置为背景
            label_rot: numpy数组, 形状为(num_obj+1)*9, 0号位置为背景
            label_vs: numpy数组, 形状为(num_obj+1,), 0号位置为背景
        '''
        with open(file_name,'r') as csv_file:  
            all_lines=csv.reader(csv_file) 
            list_file = [i for i in all_lines]  
        array_file = np.array(list_file)[1:] # exclude titles
        num_obj = int(array_file.shape[0])
        label_trans = array_file[:,2:5].astype('float32')
        label_rot = array_file[:, 5:14].astype('float32')
        label_id = array_file[:, 1:2].astype('float32')
        label_name = array_file[:, 0]
        return label_trans, label_rot,label_id,label_name

    def individual_label_csv(self, file_name):
        # 读取单个物体的标签csv文件, 返回float32类型的数组
        with open(file_name,'r') as csv_file:  
            all_lines=csv.reader(csv_file) 
            list_file = [i for i in all_lines]  
        array_file = np.array(list_file).astype('float32')   # exclude titles
        return array_file

    def create_mesh_cylinder(self,radius, height, R, t, collision):
        # 创建一个圆柱体网格, 并根据输入的旋转矩阵R和平移t进行变换
        # 如果collision为True, 圆柱体颜色为红色, 否则为绿色
        cylinder = o3d.geometry.TriangleMesh().create_cylinder(radius, height)
        vertices = np.asarray(cylinder.vertices)
        vertices[:, 2] += height / 2
        vertices = np.dot(R, vertices.T).T + t
        cylinder.vertices = o3d.utility.Vector3dVector(vertices)
        # 颜色编码, collision为1时红色, 否则为0
        ball_colors = np.zeros((vertices.shape[0], 3), dtype=np.float32)
        ball_colors[:, 0] = collision
        cylinder.vertex_colors = o3d.utility.Vector3dVector(ball_colors)
        return cylinder

    def process_train_set(self,depth_img, segment_img, gt_file_path, output_file_path,individual_object_size_path,xyz_limit=None):
        # 处理单个训练样本, 生成点云、法线、吸取分数等并保存为h5
        start_time = time.time()    # 程序开始时间

        assert depth_img.shape == (self.params['resolutionY'], self.params['resolutionX']) and depth_img.dtype == np.uint16
        label_trans, label_rot, label_id,label_name = self._read_label_csv(gt_file_path)
        obj_num = label_trans.shape[0]
        # 提取深度图中非零像素的坐标和深度
        xs = np.where(depth_img != 0)[1]
        ys = np.where(depth_img != 0)[0]
        zs = depth_img[depth_img != 0]
        # 将深度图转换为点云
        points = self._depth_to_pointcloud_optimized(xs, ys, zs, to_mm=False, xyz_limit=xyz_limit)
        # 获取前景点的物体ID
        obj_ids = np.round(segment_img[:,:, 1][segment_img[:,:, 2] == 1] * (obj_num - 1)).astype('int')

        num_pnt = points.shape[0]
        if num_pnt == 0:
            print('No foreground points!!!!!')
            return
        # 若点数不足目标点数, 则复制补齐
        if num_pnt <= self.target_num_point:
            t = int(1.0 * self.target_num_point / num_pnt) + 1
            points_tile = np.tile(points, [t, 1])
            points = points_tile[:self.target_num_point]
            obj_ids_tile = np.tile(obj_ids, [t])
            obj_ids = obj_ids_tile[:self.target_num_point]
        # 若点数超出目标点数, 则进行最远点采样
        if num_pnt > self.target_num_point:
            points_transpose = torch.from_numpy(points.reshape(1, points.shape[0], points.shape[1])).float()
            points_transpose = points_transpose.cuda()
            sampled_idx =  furthest_point_sample(points_transpose, self.target_num_point).cpu().numpy().reshape(self.target_num_point)
            points = points[sampled_idx]
            obj_ids = obj_ids[sampled_idx]

        # 读取单个物体的尺寸标签
        individual_object_size_lable = self.individual_label_csv(individual_object_size_path)[0]
        # 根据采样点的物体ID, 获取对应的标签
        label_trans = label_trans[obj_ids]
        label_rot = label_rot[obj_ids]
        label_id = label_id[obj_ids]
        individual_object_size_lable = individual_object_size_lable[obj_ids]

        # 构建点云对象
        pc_o3d = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))
        # 法线估计, 使用半径搜索
        pc_o3d.estimate_normals(o3d.geometry.KDTreeSearchParamRadius(0.015), fast_normal_computation=False)
        # 法线方向统一指向负Z轴
        pc_o3d.orient_normals_to_align_with_direction(np.array([0., 0., -1.]))
        pc_o3d.normalize_normals()
        suction_points = points
        suction_or=np.array(pc_o3d.normals).astype(np.float32)

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

        # 坐标归一化到物体坐标系
        suction_points_normalization = np.matmul((suction_points - label_trans).reshape(16384,1,3), label_rot.reshape(-1,3,3) )
        suction_points_normalization = suction_points_normalization.reshape(16384, 3)
        suction_seal_scores  = np.zeros((16384, ))
        # 计算吸取点的密封分数
        for index in range(len(label_name)):
            # 读取每个物体的稀疏点和分数
            annotation = np.load(os.path.join(OBJ_PATH, label_name[index], "labels.npz"))
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

        # 可视化吸取分数
        show_point_temp=o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))
        colors_temp = [[0, 0, 1]  for i in range(points.shape[0])]
        show_point_temp.colors = o3d.utility.Vector3dVector(colors_temp)
        vis_list = [  show_point_temp   ]
        for idx in range(len(suction_points[0:1024*4])):
            suction_point = suction_points[idx]
            suction_score = suction_seal_scores[idx]
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
            suction_score = suction_seal_scores[idx]
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
        plt.hist(suction_seal_scores, bins=100)
        plt.title("Histogram of Data")
        plt.xlabel("Value")
        plt.ylabel("Frequency")
        plt.show()

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

        # 计算吸取点的可行性分数(碰撞检测)
        height = 0.1
        radius = 0.01
        scence_point = points
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

        # 综合所有分数, 得到最终分数并排序
        score_all = suction_seal_scores * suction_wrench_scores
        score_all = suction_seal_scores * suction_wrench_scores* suction_feasibility_scores*individual_object_size_lable
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
            f['suction_seal_scores'] = suction_seal_scores
            f['suction_wrench_scores'] = suction_wrench_scores
            f['suction_feasibility_scores'] = suction_feasibility_scores
            f['individual_object_size_lable'] = individual_object_size_lable










