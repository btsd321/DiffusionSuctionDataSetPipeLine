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
        self._input_normals_ = None
        self._input_normals_before_flip = None
        self._input_packages = None

        # 输出参数定义,一维形状必须是self._output_points_num
        self._output_pointcloud = None
        self._output_normals_before_flip = None
        self._output_opposite_direction_mask = None
        self._output_visibility = None
        self._wrench_scores = None  # 扭矩评分

    def run(self):
        print("开始计算标签数据...")
        if len(self._point_cloud, axis=0) < self._output_points_num:
            # 先计算分数再上采样
            self._input_pointcloud = self._input_loader.get_point_cloud()
            self._input_normals_ = self._input_loader.get_normals()
            self._input_normals_before_flip = self._input_loader.get_normals_before_flip()
            self._input_packages = self._input_loader.get_packages()

            origin_wrench_scores = self._cal_wrench_scores()
            origin_opposite_direction_mask = self._cal_opposite_direction_mask()
            origin_visibility = self._cal_visibility()
            # 对点云及分数进行上采样
            # 计算重复次数
            t = int(1.0 * self._output_points_num / len(self._point_cloud, axis=0)) + 1
            points_tile = np.tile(self._input_pointcloud, [t, 1])
            self._output_pointcloud = points_tile[:self._output_points_num]
            normals_before_flip_tile = np.tile(self._input_normals_before_flip, [t, 1])
            self._output_normals_before_flip = normals_before_flip_tile[:self._output_points_num]
            # normals_tile = np.tile(self._input_normals_, [t, 1])
            # self._output_normals = normals_tile[:self._output_points_num]
            wrench_scores_tile = np.tile(origin_wrench_scores, [t, 1])
            self._wrench_scores = wrench_scores_tile[:self._output_points_num]
            opposite_direction_mask_tile = np.tile(origin_opposite_direction_mask, [t, 1])
            self._output_opposite_direction_mask = opposite_direction_mask_tile[:self._output_points_num]
            visibility_tile = np.tile(origin_visibility, [t, 1])
            self._output_visibility = visibility_tile[:self._output_points_num]
        elif len(self._point_cloud, axis=0) == self._output_points_num:
            # 直接计算分数
            self._wrench_scores = self._cal_wrench_scores()
            self._output_visibility = self._cal_visibility()
            self._output_opposite_direction_mask = self._cal_opposite_direction_mask()
            self._output_pointcloud = self._input_loader.get_point_cloud()
            self._output_normals_before_flip = self._input_loader.get_normals_before_flip()
        else:
            # 先下采样再计算分数
            self._input_loader.downsample(self._output_points_num)
            self._input_pointcloud = self._input_loader.get_point_cloud()
            self._input_normals_ = self._input_loader.get_normals()
            self._input_normals_before_flip = self._input_loader.get_normals_before_flip()
            self._input_packages = self._input_loader.get_packages()

            self._wrench_scores = self._cal_wrench_scores()
            self._output_opposite_direction_mask = self._cal_opposite_direction_mask()
            self._output_visibility = self._cal_visibility()
        print("标签数据计算完成。")

    def save_to_h5(self, save_path:str = None):
        if save_path is None:
            save_path = self._output_file_path
        else:
            if os.path.dirname(save_path) != "" and not os.path.exists(os.path.dirname(save_path)):
                os.makedirs(os.path.dirname(save_path))
        with h5py.File(save_path,'w') as f:
            f['points'] = self._output_pointcloud  # 使用处理后的点云
            f['suction_or'] = self._input_normals_before_flip  # 输入翻转前的法向量
            f['normal_flip_mask'] = self._output_opposite_direction_mask  # 法向量翻转掩码, 需要训练
            f['suction_wrench_scores'] = self._wrench_scores
            # f['suction_feasibility_scores'] = score_collision
            f['visibility_scores'] = self._output_visibility

    def _cal_opposite_direction_mask(self):
        """
        计算场景中物体的反向法向量掩码
        返回:
            np.ndarray: 反向法向量掩码
        """
        if self._input_normals_ is None:
            return None
        # 计算反向法向量掩码
        return self._input_loader.get_opposite_direction_mask().astype(np.float32)

    def _cal_visibility(self):
        """
        计算场景中物体的可见性
        返回:
            np.ndarray: 可见性掩码
        """
        if self._input_normals_ is None:
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
        scores = np.zeros(len(self._input_pointcloud), dtype=np.float64) # 形状（N,）
        gravity = np.array([[0, 0, -1]], dtype=np.float64) * 9.8  # 重力方向
        try:
            for obj_id, pkg in self._input_packages.items():
                # 提取物体信息
                obj_points = pkg.get_points().astype(np.float64)
                obj_normals = pkg.get_normals().astype(np.float64)
                obj_center = pkg.get_center().astype(np.float64)
                obj_mask_in_scene = pkg.get_mask_in_scene()
                # 生成一个法向量的旋转矩阵（世界坐标系到吸盘局部坐标系）
                obj_normals_rotation_matrix = _viewpoint_to_matrix_z(obj_normals)
                # 计算重力臂
                gravity_arm = (obj_center - obj_points)[np.newaxis, :]
                # 将力臂向量从世界坐标系转换到吸盘局部坐标系
                gravity_arm_local = np.matmul(gravity_arm, obj_normals_rotation_matrix)
                # 将重力向量从世界坐标系转换到吸盘局部坐标系
                gravity_local = np.matmul(gravity, obj_normals_rotation_matrix)
                # 计算吸盘坐标系下重力的扭矩
                torque_y = gravity_local[0, 0] * gravity_arm_local[0, 2] - gravity_local[0, 2] * gravity_arm_local[0, 0]
                torque_x = -gravity_local[0, 1] * gravity_arm_local[0, 2] + gravity_local[0, 2] * gravity_arm_local[0, 1]
                torque = np.sqrt(torque_x**2 + torque_y**2)
                scores[obj_mask_in_scene] = 1 - min(1, torque / wrench_thre)

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