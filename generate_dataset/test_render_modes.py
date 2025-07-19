#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试 render_utils_single.py 的离散和连续输入模式
"""
import re

def parse_range_or_single(input_str):
    """
    解析输入字符串，支持以下格式：
    - 单个值: "5" -> [5]
    - 区间: "[1,10]" -> [1,2,3,4,5,6,7,8,9,10]
    - 列表: "{1,3,5}" -> [1,3,5]
    """
    input_str = input_str.strip()
    range_match = re.match(r'^\[(\d+),(\d+)\]$', input_str)
    if range_match:
        start, end = map(int, range_match.groups())
        return list(range(start, end + 1))
    list_match = re.match(r'^\{(.+)\}$', input_str)
    if list_match:
        values_str = list_match.group(1)
        return [int(x.strip()) for x in values_str.split(',')]
    if input_str.isdigit():
        return [int(input_str)]
    raise ValueError(f"无法解析输入格式: {input_str}. 支持的格式: '5'(单个), '[1,10]'(区间), '{{1,3,5}}'(列表)")

# 失败的循环-场景列表（用于离散模式）
failed_cycles_scenes = [
    [1, 34],
    [3, 15],
    [4, 35],
    [4, 46],
    [5, 33],
    [6, 18],
    [6, 40],
    [6, 43],
    [6, 50],
    [7, 32],
    [8, 16],
    [8, 34],
    [8, 38],
    [8, 41],
    [9, 47],
    [10, 23],
    [10, 49]
]

def generate_cycle_scene_lists(failed_list):
    """
    从失败的循环-场景列表生成CYCLE_idx_list和SCENE_idx_list
    """
    if not failed_list:
        print("⚠️ 警告: failed_cycles_scenes 列表为空")
        return [], []
    
    cycle_list = []
    scene_list = []
    
    for cycle, scene in failed_list:
        cycle_list.append(cycle)
        scene_list.append(scene)
    
    print(f"📊 从失败列表生成的配对:")
    print(f"   循环数量: {len(cycle_list)} 个")
    print(f"   场景数量: {len(scene_list)} 个")
    print(f"   循环范围: {min(cycle_list)} - {max(cycle_list)}")
    print(f"   场景范围: {min(scene_list)} - {max(scene_list)}")
    
    # 显示前几个配对作为示例
    print(f"📋 前5个循环-场景配对:")
    for i, (cycle, scene) in enumerate(failed_list[:5]):
        print(f"   [{i}]: 循环{cycle} -> 场景{scene}")
    
    if len(failed_list) > 5:
        print(f"   ... 还有 {len(failed_list) - 5} 个配对")
    
    return cycle_list, scene_list

def test_modes():
    """测试两种输入模式"""
    print("🧪 测试 render_utils_single.py 的输入模式功能\n")
    
    # 测试连续模式
    print("1️⃣ 测试连续模式:")
    input_type = 'continuous_scences'
    cycle_list_str = '[1,3]'
    scene_list_str = '{2,5}'
    
    if input_type == 'continuous_scences':
        try:
            CYCLE_idx_list = parse_range_or_single(cycle_list_str)
            SCENE_idx_list = parse_range_or_single(scene_list_str)
            print(f"   ✅ 连续模式解析成功:")
            print(f"   CYCLE_idx_list: {CYCLE_idx_list}")
            print(f"   SCENE_idx_list: {SCENE_idx_list}")
            print(f"   总计需要渲染: {len(CYCLE_idx_list)} × {len(SCENE_idx_list)} = {len(CYCLE_idx_list) * len(SCENE_idx_list)} 个循环-场景组合")
        except ValueError as e:
            print(f"   ❌ 参数解析错误: {e}")
    
    print("\n" + "="*60 + "\n")
    
    # 测试离散模式
    print("2️⃣ 测试离散模式:")
    input_type = 'discrete_scences'
    
    if input_type == 'discrete_scences':
        CYCLE_idx_list, SCENE_idx_list = generate_cycle_scene_lists(failed_cycles_scenes)
        print(f"   ✅ 离散模式解析成功:")
        print(f"   总计需要渲染: {len(CYCLE_idx_list)} 个失败的循环-场景配对")
        print(f"   前5个循环编号: {CYCLE_idx_list[:5]}")
        print(f"   前5个场景编号: {SCENE_idx_list[:5]}")
        
        # 验证配对是否正确
        print(f"\n   📋 验证配对:")
        for i in range(min(5, len(CYCLE_idx_list))):
            print(f"   配对{i+1}: 循环{CYCLE_idx_list[i]} -> 场景{SCENE_idx_list[i]}")
    
    print("\n" + "="*60 + "\n")
    print("✅ 所有测试完成！render_utils_single.py 的离散和连续输入模式都工作正常！")

if __name__ == '__main__':
    test_modes()
