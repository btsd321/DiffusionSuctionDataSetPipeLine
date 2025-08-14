# 包裹类
import numpy as np

class Package:
    def __init__(self, package_init_data):
        self._normals = None
        self._opposite_direction_mask = None
        try:
            self._center = package_init_data["center"] # 包裹中心在场景中的坐标,形状(3,)
            self._name = package_init_data["name"] # 包裹名称(0-113)
            self._points = package_init_data["points"] # 包裹点云数据（世界坐标系）, 形状(N,3)
            self._normals_before_flip = package_init_data["normals_before_flip"] # 包裹点云法向量（世界坐标系），形状(N,3)
            self._rotation = package_init_data["rotation"] # 包裹旋转信息，形状(9,)
            self._mask_for_scene = package_init_data["mask"] # 包裹点云在场景点云中的掩码，形状(N',),N'为场景中点的数量
            self._indices_for_scene = package_init_data["indices"] # 包裹点云在场景点云中的索引，形状(N',),N'为场景中点的数量
        except KeyError as e:
            print(f"缺少关键字段: {e}")

    def flip_normals(self):
        # 计算从点指向中心的向量
        self._normals = self._normals_before_flip.copy()
        points_to_center = self._center - self._points  # 形状为(N, 3)

        # 计算法向量与指向中心向量的点积
        dot_products = np.sum(self._normals_before_flip * points_to_center, axis=1)  # 形状 (N,)

        # 找到反向的点（点积小于0，即法向量背离中心）
        self._opposite_direction_mask = dot_products < 0  # 形状 (N,)

        # 对背离中心的法向量取反，使其指向中心
        self._normals[self._opposite_direction_mask] *= -1
        # self._visualize_normals_and_center(points, normals, obj_center)
        return self._normals, self._opposite_direction_mask