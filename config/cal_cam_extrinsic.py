import numpy as np
from scipy.spatial.transform import Rotation

# Blender中验证过的正确参数
# 坐标系类型：Z+为上方向，X+为右方向，Y+为前方向
# 相机朝向指的是相机坐标系Z轴负方向
# 相机位置
cam_pos = np.array([0.0, 0.5, 0.9])  # 相机位置
# 相机朝向的点
target = np.array([0, -0.1, 0])      # 目标点（用于验证）

# 计算相机朝向向量（相机坐标系Z轴负方向）
forward = target - cam_pos
forward = forward / np.linalg.norm(forward)  # 归一化

# 定义世界坐标系的上方向（Z+）
world_up = np.array([0, 0, 1])

# 计算相机坐标系的右方向（X轴）
right = np.cross(forward, world_up)
right = right / np.linalg.norm(right)

# 计算相机坐标系的上方向（Y轴）
up = np.cross(right, forward)

# 构建相机坐标系到世界坐标系的旋转矩阵
# 相机坐标系：X轴为right，Y轴为up，Z轴为-forward（相机朝向为Z轴负方向）
R_cam_to_world = np.column_stack([right, up, -forward])

# 计算相机坐标系到世界坐标系的旋转四元数，格式wxyz
rotation_cam_to_world = Rotation.from_matrix(R_cam_to_world)
quat_cam_to_world = rotation_cam_to_world.as_quat()  # scipy返回xyzw格式
quat_cam_to_world_wxyz = np.array([quat_cam_to_world[3], quat_cam_to_world[0], 
                                   quat_cam_to_world[1], quat_cam_to_world[2]])

# 计算世界坐标系到相机坐标系的旋转四元数，格式wxyz
R_world_to_cam = R_cam_to_world.T
rotation_world_to_cam = Rotation.from_matrix(R_world_to_cam)
quat_world_to_cam = rotation_world_to_cam.as_quat()  # scipy返回xyzw格式
quat_world_to_cam_wxyz = np.array([quat_world_to_cam[3], quat_world_to_cam[0], 
                                   quat_world_to_cam[1], quat_world_to_cam[2]])

# 计算相机坐标系到世界坐标系的齐次变换矩阵
T_cam_to_world = np.eye(4)
T_cam_to_world[:3, :3] = R_cam_to_world
T_cam_to_world[:3, 3] = cam_pos

# 计算世界坐标系到相机坐标系的齐次变换矩阵
T_world_to_cam = np.eye(4)
T_world_to_cam[:3, :3] = R_world_to_cam
T_world_to_cam[:3, 3] = -R_world_to_cam @ cam_pos

# 打印结果
print("相机位置:", cam_pos)
print("目标点:", target)
print("相机朝向向量:", forward)
print("\n相机坐标系到世界坐标系的旋转四元数 (wxyz):")
print(quat_cam_to_world_wxyz)
print("\n世界坐标系到相机坐标系的旋转四元数 (wxyz):")
print(quat_world_to_cam_wxyz)
print("\n相机坐标系到世界坐标系的齐次变换矩阵:")
print(T_cam_to_world)
print("\n世界坐标系到相机坐标系的齐次变换矩阵:")
print(T_world_to_cam)

# 验证：将目标点从世界坐标系转换到相机坐标系
target_homogeneous = np.append(target, 1)
target_in_cam = T_world_to_cam @ target_homogeneous
print(f"\n验证：目标点在相机坐标系中的位置: {target_in_cam[:3]}")
print(f"目标点在相机坐标系中应该在Z轴负方向，Z值应该为负: {target_in_cam[2] < 0}")

print("\n验证: 相机坐标系下点(0.0, 0.08320503, 1.02619536)在世界坐标系下的坐标:")
point_in_world = T_cam_to_world @ np.array([0.0, -0.08320503, -1.02619536, 1.0])
print(f'世界坐标系下坐标为：', point_in_world[:3])
print("\n验证: 相机坐标系下点(0.0, 1, 0)在世界坐标系下的坐标:")
point_in_world = T_cam_to_world @ np.array([0.0, 1, 0, 1.0])
print(f'世界坐标系下坐标为：', point_in_world[:3])
print("\n验证: 相机坐标系下点(1, 0, 0)在世界坐标系下的坐标:")
point_in_world = T_cam_to_world @ np.array([1, 0, 0, 1.0])
print(f'世界坐标系下坐标为：', point_in_world[:3])
print("\n验证: 世界坐标系下点(0.05, -0.04, 0.05)在相机坐标系下的坐标:")
point_in_camera = T_world_to_cam @ np.array([0.05, -0.04, 0.05, 1.0])
print(f'相机坐标系下坐标为：', point_in_camera[:3])
