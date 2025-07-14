import numpy as np
from scipy.spatial.transform import Rotation as R

R_mat = np.array([
    [1.0, 0.0, 0.0],
    [0.0, -0.87415728, -0.48564293],
    [-0.0, 0.48564293, -0.87415728]
])

r = R.from_matrix(R_mat)
quat = r.as_quat()  # [x, y, z, w] 格式
print(quat)