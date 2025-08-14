# 标注评分计算方法
import scene_loader
import package
import numpy as np

def cal_wrench_scores(scene: scene_loader.SceneLoader):
    """
    计算场景中物体的扭矩评分

    参数:
        scene (scene_loader.SceneLoader): 场景加载器实例

    返回:
        float: 计算得到的扭矩评分
    """
    total_torque = 0.0
    for obj_id, pkg in scene._packages.items():
        # 计算每个包裹的扭矩
        torque = package.calculate_torque(pkg)
        total_torque += torque
    return total_torque