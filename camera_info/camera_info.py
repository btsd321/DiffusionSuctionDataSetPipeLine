import os
import yaml

class CameraInfo:
    def __init__(self, file_path):
        # 判断文件是否存在
        if not os.path.exists(file_path):
            raise FileNotFoundError("Camera info file not found.")
        # 读取yaml文件
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                self.camera_info = yaml.load(f, Loader=yaml.FullLoader)
        except yaml.YAMLError as e:
            raise ValueError(f"YAML文件解析失败: {e}\n请检查文件格式是否正确: {file_path}")

        # 直接作为属性暴露常用参数
        self.focal_length = self.camera_info.get('focal_length')
        self.sensor_size = self.camera_info.get('sensor_size')
        self.cam_translation_vector = self.camera_info.get('cam_translation_vector')
        self.cam_quaternions = self.camera_info.get('cam_quaternions')
        self.extrinsic_matrix = self._to_matrix(self.camera_info.get('extrinsic_matrix'))
        self.intrinsic_matrix = self._to_matrix(self.camera_info.get('intrinsic_matrix'))

    def _to_matrix(self, mat):
        """
        将嵌套list转为numpy矩阵
        """
        import numpy as np
        if mat is None:
            return None
        return np.array(mat, dtype=float)

    def get_extrinsic(self):
        """返回外参矩阵 (4x4)"""
        return self.extrinsic_matrix

    def get_intrinsic(self):
        """返回内参矩阵 (3x3)"""
        return self.intrinsic_matrix

    def get_translation(self):
        """返回相机平移向量 (1x3)"""
        return self.cam_translation_vector

    def get_quaternion(self):
        """返回相机四元数旋转 (1x4)"""
        return self.cam_quaternions

    def __repr__(self):
        return f"CameraInfo(focal_length={self.focal_length}, sensor_size={self.sensor_size}, translation={self.cam_translation_vector}, quaternion={self.cam_quaternions})"

def get_camera_info_from_yaml(file_path):
    """
    获取相机信息
    :param file_path: 相机信息文件路径
    :return: CameraInfo对象
    """
    return CameraInfo(file_path)


if __name__ == '__main__':
    file_path = r"D:/Project/DiffusionSuctionDataSetPipeLine/camera_info/camera_info.yaml"
    camera_info = get_camera_info_from_yaml(file_path)
    print(camera_info)