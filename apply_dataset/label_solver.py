# 标注评分计算方法
import scene_loader
import package
import numpy as np
import common_info
import os
import h5py

def _viewpoint_to_matrix_z(towards):
    """
    根据朝向向量生成以z轴为主的旋转矩阵
    
    用于将法向量转换为3x3旋转矩阵，其中z轴为法向量方向。
    主要用于抗扭矩计算中的坐标系变换。
    
    参数:
        towards (numpy.ndarray): 法向量，形状为(N, 3)或(3,)
    
    返回:
        numpy.ndarray: 旋转矩阵，形状为(N, 3, 3)或(3, 3)
    """
    # 检查输入维度
    if towards.ndim == 1:
        # 单个向量输入，直接处理
        n = towards  # shape: (3,)
        # 设置z轴为法向量方向，并标准化
        new_z = n / np.linalg.norm(n)  # shape: (3,)
        # 构造y轴，垂直于z轴在水平面内
        new_y = np.array([new_z[1], -new_z[0], 0], dtype=np.float64)
        # 处理法向量接近z轴的特殊情况
        if np.linalg.norm(new_y) < 1e-6:
            new_y = np.array([0, 1, 0], dtype=np.float64)
        new_y = new_y / np.linalg.norm(new_y)
        # 通过叉积计算x轴
        new_x = np.cross(new_y, new_z)
        new_x = new_x / np.linalg.norm(new_x)
        # 组合成旋转矩阵 [x, y, z] 作为列向量
        rot_matrix = np.stack([new_x, new_y, new_z], axis=1)  # shape: (3, 3)
        return rot_matrix
    else:
        # 批量处理多个向量
        n = towards  # shape: (N, 3)
        # 设置z轴为法向量方向，并标准化
        new_z = n / np.linalg.norm(n, axis=1, keepdims=True)  # shape: (N, 3)
        # 构造y轴，垂直于z轴在水平面内
        new_y = np.stack([new_z[:, 1], -new_z[:, 0], np.zeros(new_z.shape[0])], axis=1)  # shape: (N, 3)
        # 处理法向量接近z轴的特殊情况
        y_norms = np.linalg.norm(new_y, axis=1, keepdims=True)  # shape: (N, 1)
        near_z_mask = (y_norms.flatten() < 1e-6)  # shape: (N,)
        # 对于接近z轴的向量，使用 [0, 1, 0] 作为y轴
        new_y[near_z_mask] = np.array([0, 1, 0])
        # 重新计算法向量（对于修改过的向量）
        y_norms = np.linalg.norm(new_y, axis=1, keepdims=True)
        new_y = new_y / y_norms  # shape: (N, 3)
        # 通过叉积计算x轴
        new_x = np.cross(new_y, new_z)  # shape: (N, 3)
        new_x = new_x / np.linalg.norm(new_x, axis=1, keepdims=True)  # shape: (N, 3)
        # 组合成旋转矩阵 [x, y, z] 作为列向量
        rot_matrix = np.stack([new_x, new_y, new_z], axis=2)  # shape: (N, 3, 3)
        return rot_matrix
    
def _viewpoint_to_matrix_x(towards):
    """
    根据朝向向量生成以x轴为主的旋转矩阵
    
    用于将法向量转换为3x3旋转矩阵，其中x轴为法向量方向。
    主要用于可行性评分计算中的坐标系变换。
    
    参数:
        towards (numpy.ndarray): 法向量，形状为(N, 3)或(3,)
    
    返回:
        numpy.ndarray: 旋转矩阵，形状为(N, 3, 3)或(3, 3)
    """
    # 检查输入维度
    if towards.ndim == 1:
        # 单个向量输入，直接处理
        n = towards  # shape: (3,)
        # 设置x轴为法向量方向，并标准化
        new_x = n / np.linalg.norm(n)  # shape: (3,)
        # 构造z轴，垂直于x轴在水平面内
        new_z = np.array([0, -new_x[2], new_x[1]], dtype=np.float64)
        # 处理法向量接近x轴的特殊情况
        if np.linalg.norm(new_z) < 1e-6:
            new_z = np.array([0, 0, 1], dtype=np.float64)
        new_z = new_z / np.linalg.norm(new_z)
        # 通过叉积计算y轴
        new_y = np.cross(new_z, new_x)
        new_y = new_y / np.linalg.norm(new_y)
        # 组合成旋转矩阵 [x, y, z] 作为列向量
        rot_matrix = np.stack([new_x, new_y, new_z], axis=1)  # shape: (3, 3)
        return rot_matrix
    else:
        # 批量处理多个向量
        n = towards  # shape: (N, 3)
        # 设置x轴为法向量方向，并标准化
        new_x = n / np.linalg.norm(n, axis=1, keepdims=True)  # shape: (N, 3)
        # 构造z轴，垂直于x轴在水平面内
        new_z = np.stack([np.zeros(new_x.shape[0]), -new_x[:, 2], new_x[:, 1]], axis=1)  # shape: (N, 3)
        # 处理法向量接近x轴的特殊情况
        z_norms = np.linalg.norm(new_z, axis=1, keepdims=True)  # shape: (N, 1)
        near_x_mask = (z_norms.flatten() < 1e-6)  # shape: (N,)
        # 对于接近x轴的向量，使用 [0, 0, 1] 作为z轴
        new_z[near_x_mask] = np.array([0, 0, 1])
        # 重新计算法向量（对于修改过的向量）
        z_norms = np.linalg.norm(new_z, axis=1, keepdims=True)
        new_z = new_z / z_norms  # shape: (N, 3)
        # 通过叉积计算y轴
        new_y = np.cross(new_z, new_x)  # shape: (N, 3)
        new_y = new_y / np.linalg.norm(new_y, axis=1, keepdims=True)  # shape: (N, 3)
        # 组合成旋转矩阵 [x, y, z] 作为列向量
        rot_matrix = np.stack([new_x, new_y, new_z], axis=2)  # shape: (N, 3, 3)
        return rot_matrix

class LabelSolver:
    def __init__(self, loader: scene_loader.SceneLoader, common_info: common_info.CommonInfo = None, output_points_num: int = 16384, output_file_path:str = None):
        self._common_info = common_info
        self._output_points_num = output_points_num
        self._output_file_path = None
        self._input_loader = loader
        if output_file_path is None:
            self._output_file_path = "output.h5"  # 默认输出路径
        else:
            # 分割路径，判断文件路径所在的文件夹是否存在
            output_dir = os.path.dirname(output_file_path)
            if output_dir != "" and not os.path.exists(output_dir):
                os.makedirs(output_dir)
            self._output_file_path = output_file_path

        # 输入参数定义
        self._input_pointcloud = None
        self._input_normals = None
        self._input_normals_before_flip = None
        self._input_packages = None

        # 输出参数定义,一维形状必须是self._output_points_num
        self._output_pointcloud = None
        self._output_normals_before_flip = None
        self._output_opposite_direction_mask = None
        self._output_visibility = None
        self._output_feasibility_scores = None
        self._output_wrench_scores = None  # 扭矩评分

    def run(self):
        input_point_cloud_num = self._input_loader.get_pointcloud().shape[0]
        if input_point_cloud_num < self._output_points_num:
            # 先计算分数再上采样
            self._input_pointcloud = self._input_loader.get_pointcloud()
            self._input_normals = self._input_loader.get_normals()
            self._input_normals_before_flip = self._input_loader.get_normals_before_flip()
            self._input_packages = self._input_loader.get_packages()

            origin_wrench_scores = self._cal_wrench_scores()
            origin_opposite_direction_mask = self._cal_opposite_direction_mask()
            origin_visibility = self._cal_visibility_scores()
            origin_feasibility_scores = self._cal_feasibility_scores()
            # 对点云及分数进行上采样
            # 计算重复次数
            t = int(1.0 * self._output_points_num / self._input_pointcloud.shape[0]) + 1
            points_tile = np.tile(self._input_pointcloud, [t, 1])
            self._output_pointcloud = points_tile[:self._output_points_num]
            normals_before_flip_tile = np.tile(self._input_normals_before_flip, [t, 1])
            self._output_normals_before_flip = normals_before_flip_tile[:self._output_points_num]
            # normals_tile = np.tile(self._input_normals, [t, 1])
            # self._output_normals = normals_tile[:self._output_points_num]
            wrench_scores_tile = np.tile(origin_wrench_scores, t)
            self._output_wrench_scores = wrench_scores_tile[:self._output_points_num]
            opposite_direction_mask_tile = np.tile(origin_opposite_direction_mask, t)
            self._output_opposite_direction_mask = opposite_direction_mask_tile[:self._output_points_num]
            visibility_tile = np.tile(origin_visibility, t)
            self._output_visibility = visibility_tile[:self._output_points_num]
            feasibility_scores_tile = np.tile(origin_feasibility_scores, t)
            self._output_feasibility_scores = feasibility_scores_tile[:self._output_points_num]
        elif input_point_cloud_num == self._output_points_num:
            # 直接计算分数
            self._output_wrench_scores = self._cal_wrench_scores()
            self._output_visibility = self._cal_visibility_scores()
            self._output_feasibility_scores = self._cal_feasibility_scores()
            self._output_opposite_direction_mask = self._cal_opposite_direction_mask()
            self._output_pointcloud = self._input_loader.get_pointcloud()
            self._output_normals_before_flip = self._input_loader.get_normals_before_flip()
        else:
            # 先下采样再计算分数
            self._input_loader.downsample(self._output_points_num)
            self._input_pointcloud = self._input_loader.get_pointcloud()
            self._input_normals = self._input_loader.get_normals()
            self._input_normals_before_flip = self._input_loader.get_normals_before_flip()
            self._input_packages = self._input_loader.get_packages()
            
            self._output_pointcloud = self._input_pointcloud
            self._output_normals_before_flip = self._input_normals_before_flip
            self._output_wrench_scores = self._cal_wrench_scores()
            self._output_opposite_direction_mask = self._cal_opposite_direction_mask()
            self._output_visibility = self._cal_visibility_scores()
            self._output_feasibility_scores = self._cal_feasibility_scores()

    def save_to_h5(self, save_path:str = None):
        if save_path is None:
            save_path = self._output_file_path
        else:
            if os.path.dirname(save_path) != "" and not os.path.exists(os.path.dirname(save_path)):
                os.makedirs(os.path.dirname(save_path))
        with h5py.File(save_path,'w') as f:
            f['points'] = self._output_pointcloud  # 使用处理后的点云
            f['normals'] = self._output_normals_before_flip  # 输入翻转前的法向量
            f['normal_flip_mask'] = self._output_opposite_direction_mask  # 法向量翻转掩码, 需要训练
            f['wrench_scores'] = self._output_wrench_scores
            f['feasibility_scores'] = self._output_feasibility_scores
            f['visibility_scores'] = self._output_visibility

    def _cal_opposite_direction_mask(self):
        """
        计算场景中物体的反向法向量掩码
        返回:
            np.ndarray: 反向法向量掩码
        """
        if self._input_normals is None:
            return None
        # 计算反向法向量掩码
        return self._input_loader.get_opposite_direction_mask().astype(np.float32)

    def _cal_visibility_scores(self):
        """
        计算场景中物体的可见性
        返回:
            np.ndarray: 可见性掩码
        """
        if self._input_normals is None:
            return None
        # 计算可见性掩码
        return self._input_loader.get_visibility().astype(np.float32)

    def _cal_wrench_scores(self):
        """
        计算场景中物体的扭矩评分
        返回:
            float: 计算得到的扭矩评分
        """

        # 参数初始化
        k = 30          # 吸盘刚度系数
        radius = 0.01   # 吸盘半径
        wrench_thre = k * radius * np.pi * np.sqrt(2)
        scores = np.zeros(self._input_pointcloud.shape[0], dtype=np.float64) # 形状（N,）
        gravity = np.array([[0, 0, -1]], dtype=np.float64) * 9.8  # 重力方向
        try:
            for obj_id, pkg in self._input_packages.items():
                # 提取物体信息
                obj_points = pkg.get_pointcloud().astype(np.float64)    # 形状(N', 3), N'为物体点云中的点数
                obj_normals = pkg.get_normals().astype(np.float64)      # 形状(N', 3), N'为物体点云中的点数
                obj_center = pkg.get_center().astype(np.float64)        # 形状(3,)
                obj_mask_in_scene = pkg.get_mask_in_scene()             # 形状(N,)
                # 生成一个法向量的旋转矩阵（世界坐标系到吸盘局部坐标系）
                obj_normals_rotation_matrix = _viewpoint_to_matrix_z(obj_normals) # shape: (N', 3, 3)
                # 计算重力臂
                gravity_arm = obj_center[np.newaxis, :] - obj_points  # shape: (N', 3)
                # 将力臂向量从世界坐标系转换到吸盘局部坐标系
                gravity_arm_local = np.einsum('nij,nj->ni', obj_normals_rotation_matrix, gravity_arm)  # shape: (N', 3)
                # 将重力向量从世界坐标系转换到吸盘局部坐标系
                gravity_local = np.einsum('nij,j->ni', obj_normals_rotation_matrix, gravity[0])  # shape: (N', 3)
                # 计算吸盘坐标系下重力的扭矩
                torque_x = gravity_arm_local[:, 1] * gravity_local[:, 2] - gravity_arm_local[:, 2] * gravity_local[:, 1]
                torque_y = gravity_arm_local[:, 2] * gravity_local[:, 0] - gravity_arm_local[:, 0] * gravity_local[:, 2]
                torque = np.sqrt(torque_x**2 + torque_y**2)
                scores[obj_mask_in_scene] = 1 - np.minimum(1.0, torque / wrench_thre)

            # 定义参考向量，指向负Z轴方向（垂直向下，符合重力方向）
            reference_vector = np.array([0, 0, -1]).astype(np.float64)
            # 计算法向量与参考向量的点积（向量内积）
            dot_products = np.sum(self._input_normals * reference_vector, axis=1)
            # 计算每个法向量的模长（向量长度）
            norm_magnitudes = np.linalg.norm(self._input_normals, axis=1)
            # 计算参考向量的模长
            reference_magnitude = np.linalg.norm(reference_vector)
            # 根据向量点积公式计算夹角的余弦值: cos(θ) = (a·b) / (|a|*|b|)
            cos_angles = dot_products / (norm_magnitudes * reference_magnitude)
            # 防止数值计算误差导致余弦值超出[-1, 1]范围
            cos_angles = np.clip(cos_angles, -1.0, 1.0)
            # 通过反余弦函数计算夹角（弧度制）
            angles = np.arccos(cos_angles)
            # 将弧度转换为角度制，便于理解和调试
            angles_degrees = np.degrees(angles).astype(np.float64)
            # 将角度映射到[0,1]权重区间：
            # - 0度（完全垂直向下）→ 权重1.0
            # - 90度（水平方向）→ 权重0.5  
            # - 180度（完全垂直向上）→ 权重0.0
            mapped_values = 1 - (angles_degrees / 180)
            return scores * mapped_values
        except Exception as e:
            print(f"Error occurred while calculating wrench scores: {e}")
            raise e
    
    
    
    def _cal_feasibility_scores(self, device = 'cuda'):
        '''
        计算可行性评分(和其他物体是否有碰撞的评分)
        '''
        height = 0.15 # 吸盘高度
        radius = 0.02 # 吸盘半径
        if device == 'cpu':
            # 计算吸取点的可行性分数(碰撞检测)
            try:
                pointcloud = self._input_loader.get_pointcloud()
                scores = np.zeros(pointcloud.shape[0], dtype=np.bool)
                for index, point in enumerate(pointcloud):
                    targets = pointcloud.copy()
                    normal = self._input_loader.get_normals()[index]
                    rotation_matrix = _viewpoint_to_matrix_x(normal)
                    targets = targets - point
                    targets = np.matmul(targets, rotation_matrix)
                    targets_yz = targets[:, 1:3]
                    targets_r = np.linalg.norm(targets_yz, axis=-1)
                    mask1 = targets_r < radius
                    mask2 = ((targets[:,0] > 0.01) & (targets[:,0] < height))
                    mask = np.any(mask1 & mask2)
                    scores[index] = mask
                scores = ~np.array(scores)  # 转换为布尔类型并取反
                return scores.astype(np.float32)
            except Exception as e:
                print(f"Error occurred while calculating feasibility scores: {e}")
                raise e
        else:
            try:
                scores = self.__cal_feasibility_scores_gpu(radius = radius, height = height)
                return scores
            except Exception as e:
                print(f"Error occurred while calculating feasibility scores: {e}")
                raise e
        
        
    def __cal_feasibility_scores_gpu(self, radius=1.0, height=1.0):
        # GPU内核函数
        import math
        from numba import cuda
        if not cuda.is_available():
            raise RuntimeError("CUDA is not available. Please check your environment.")
        @cuda.jit
        def collision_kernel(pointcloud, normals, collision_detected, height, radius, N):
            idx = cuda.grid(1)
            if idx >= N:
                return
            
            # 当前点和法向量
            px, py, pz = pointcloud[idx, 0], pointcloud[idx, 1], pointcloud[idx, 2]
            nx, ny, nz = normals[idx, 0], normals[idx, 1], normals[idx, 2]
            
            # 标准化法向量
            n_norm = math.sqrt(nx*nx + ny*ny + nz*nz)
            if n_norm < 1e-10:
                collision_detected[idx] = False
                return
            nx, ny, nz = nx/n_norm, ny/n_norm, nz/n_norm
            
            # 构建旋转矩阵（_viewpoint_to_matrix_x的GPU实现）
            # 设置x轴为法向量方向
            new_x_x, new_x_y, new_x_z = nx, ny, nz
            
            # 构造z轴，垂直于x轴：[0, -new_x[2], new_x[1]]
            new_z_x = 0.0
            new_z_y = -new_x_z
            new_z_z = new_x_y
            
            # 处理法向量接近x轴的特殊情况
            z_norm = math.sqrt(new_z_x*new_z_x + new_z_y*new_z_y + new_z_z*new_z_z)
            if z_norm < 1e-6:
                # 使用 [0, 0, 1] 作为z轴
                new_z_x, new_z_y, new_z_z = 0.0, 0.0, 1.0
            else:
                # 标准化z轴
                new_z_x, new_z_y, new_z_z = new_z_x/z_norm, new_z_y/z_norm, new_z_z/z_norm
            
            # 通过叉积计算y轴：y = z × x
            new_y_x = new_z_y * new_x_z - new_z_z * new_x_y
            new_y_y = new_z_z * new_x_x - new_z_x * new_x_z
            new_y_z = new_z_x * new_x_y - new_z_y * new_x_x
            
            # 标准化y轴
            y_norm = math.sqrt(new_y_x*new_y_x + new_y_y*new_y_y + new_y_z*new_y_z)
            if y_norm < 1e-10:
                collision_detected[idx] = False
                return
            new_y_x, new_y_y, new_y_z = new_y_x/y_norm, new_y_y/y_norm, new_y_z/y_norm
            
            # 旋转矩阵 [x, y, z] 作为列向量
            # rotation_matrix = [[new_x_x, new_y_x, new_z_x],
            #                   [new_x_y, new_y_y, new_z_y],
            #                   [new_x_z, new_y_z, new_z_z]]
            
            # 检查与所有其他点的碰撞
            has_collision = False
            for i in range(N):
                if i == idx:
                    continue
                
                # 相对位置
                dx = pointcloud[i, 0] - px
                dy = pointcloud[i, 1] - py
                dz = pointcloud[i, 2] - pz
                
                # 坐标变换：targets = relative_positions @ rotation_matrix.T
                # 即 targets = rotation_matrix.T @ relative_positions
                target_x = new_x_x * dx + new_x_y * dy + new_x_z * dz
                target_y = new_y_x * dx + new_y_y * dy + new_y_z * dz
                target_z = new_z_x * dx + new_z_y * dy + new_z_z * dz
                
                # 碰撞检测逻辑（对应CPU版本）
                # targets_yz = targets[:, 1:3]  -> [target_y, target_z]
                # targets_r = np.linalg.norm(targets_yz, axis=-1)
                targets_r = math.sqrt(target_y * target_y + target_z * target_z)
                
                # mask1 = targets_r < radius
                mask1 = targets_r < radius
                
                # mask2 = ((targets[:,0] > 0.01) & (targets[:,0] < height))
                mask2 = (target_x > 0.01) and (target_x < height)
                
                # 综合判断
                if mask1 and mask2:
                    has_collision = True
                    break
            
            collision_detected[idx] = has_collision
            
        # 准备GPU数据
        d_pointcloud = cuda.to_device(self._input_loader.get_pointcloud())
        d_normals = cuda.to_device(self._input_loader.get_normals())
        N = self._input_loader.get_pointcloud().shape[0]
        d_collision_detected = cuda.device_array(N, dtype=np.bool_)
        
        # 启动GPU内核
        threads_per_block = 128
        blocks_per_grid = (N + threads_per_block - 1) // threads_per_block
        
        collision_kernel[blocks_per_grid, threads_per_block](
            d_pointcloud, d_normals, d_collision_detected, height, radius, N
        )
        
        # 复制结果回CPU
        collision_detected = d_collision_detected.copy_to_host()
        feasibility_scores = (~collision_detected).astype(np.float32)
        
        return feasibility_scores