# 包裹类
import numpy as np

class Package:
    def __init__(self, package_init_data):
        self.__normals = None
        self.__opposite_direction_mask = None
        try:
            self.__center = package_init_data["center"] # 包裹中心在场景中的坐标,形状(3,)
            self.__name = package_init_data["name"] # 包裹名称(0-113)
            self.__points = package_init_data["points"] # 包裹点云数据（世界坐标系）, 形状(N,3)
            self.__normals_before_flip = package_init_data["normals_before_flip"] # 包裹点云法向量（世界坐标系），形状(N,3)
            self.__rotation = package_init_data["rotation"] # 包裹旋转信息，形状(9,)
            self.__mask_in_scene = package_init_data["mask"] # 包裹点云在场景点云中的掩码，形状(N',),N'为场景中点的数量
            self.__indices_in_scene = package_init_data["indices"] # 包裹点云在场景点云中的索引，形状(N',),N'为场景中点的数量
            self.__visibility = package_init_data["visibility"] # 包裹可见性分数
        except KeyError as e:
            print(f"缺少关键字段: {e}")

    def flip_normals(self):
        # 计算从点指向中心的向量
        self.__normals = self.__normals_before_flip.copy()
        points_to_center = self.__center - self.__points  # 形状为(N, 3)

        # 计算法向量与指向中心向量的点积
        dot_products = np.sum(self.__normals_before_flip * points_to_center, axis=1)  # 形状 (N,)

        # 找到反向的点（点积小于0，即法向量背离中心）
        self.__opposite_direction_mask = dot_products < 0  # 形状 (N,)

        # 对背离中心的法向量取反，使其指向中心
        self.__normals[self.__opposite_direction_mask] *= -1
        # self._visualize_normals_and_center(points, normals, obj_center)
        return self.__normals, self.__opposite_direction_mask

    def get_pointcloud(self):
        return self.__points

    def get_normals(self):
        return self.__normals
    
    def get_normals_before_flip(self):
        return self.__normals_before_flip

    def get_center(self):
        return self.__center

    def get_visibility(self):
        return self.__visibility
    
    def get_name(self):
        return self.__name
    
    def get_rotation(self):
        return self.__rotation

    def get_mask_in_scene(self):
        return self.__mask_in_scene
    
    def get_indices_in_scene(self):
        return self.__indices_in_scene
    
    def get_opposite_direction_mask(self):
        return self.__opposite_direction_mask
    