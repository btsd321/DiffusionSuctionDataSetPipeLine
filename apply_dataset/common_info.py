import os
import sys
import numpy as np
import json
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'config'))
import camera_info 

class CommonInfo:
    def __init__(self, camera_info_file_path, parameter_file_path):
        self._camera_info = camera_info.get_camera_info_from_yaml(camera_info_file_path)
        self._parameters = self._load_parameters(parameter_file_path)

    def _load_parameters(self, params_file_name):
        """
        加载相机参数和处理配置文件
        
        参数:
            params_file_name (str): JSON配置文件路径
            
        返回:
            dict: 包含相机内参、深度范围等配置的字典
        """
        params = {}
        with open(params_file_name, 'r') as f:
            config = json.load(f)
            params = config
        return params 
    
    def get_camera_info(self):
        return self._camera_info
    
    def get_parameters(self):
        return self._parameters
