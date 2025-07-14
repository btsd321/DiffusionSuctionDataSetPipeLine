import numpy as np
from scipy.spatial.transform import Rotation

cam_pos = np.array([0, 0.5, 0.9])
target = np.array([0, -0.1, 0])
up = np.array([0, 0, 1])

z_axis = (cam_pos - target)
z_axis = z_axis / np.linalg.norm(z_axis)
x_axis = np.cross(up, z_axis)
x_axis = x_axis / np.linalg.norm(x_axis)
y_axis = np.cross(z_axis, x_axis)

R = np.stack([x_axis, y_axis, z_axis], axis=1)
extrinsic = np.eye(4)
extrinsic[:3, :3] = R
extrinsic[:3, 3] = cam_pos
print(extrinsic)

# 计算四元数
r = Rotation.from_matrix(R)
quat = r.as_quat()  # [x, y, z, w] 格式
print(quat)