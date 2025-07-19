#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
解析物理仿真结果检查错误日志，提取失败的循环数和场景数

@author: GitHub Copilot
"""

import os
import re
import argparse

def parse_error_log(log_file_path):
    """
    解析错误日志文件，提取失败的循环数和场景数
    
    Args:
        log_file_path (str): 日志文件路径
        
    Returns:
        list: [[cycle, scene], ...] 格式的失败列表
    """
    failed_list = []
    
    if not os.path.exists(log_file_path):
        print(f"❌ 错误：日志文件不存在: {log_file_path}")
        return failed_list
    
    # 正则表达式匹配错误行格式
    # [2025-07-19 11:12:16] 循环1-场景34-物体31: ...
    error_pattern = r'\[\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\] 循环(\d+)-场景(\d+)-物体\d+:'
    
    with open(log_file_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            
            # 跳过空行和非错误行
            if not line or not line.startswith('['):
                continue
            
            # 匹配错误行
            match = re.search(error_pattern, line)
            if match:
                cycle = int(match.group(1))
                scene = int(match.group(2))
                failed_list.append([cycle, scene])
                print(f"📍 发现错误: 循环{cycle}, 场景{scene}")
    
    return failed_list

def remove_duplicates(failed_list):
    """
    去除重复的 [cycle, scene] 组合
    
    Args:
        failed_list (list): 包含重复项的失败列表
        
    Returns:
        list: 去重后的失败列表
    """
    seen = set()
    unique_list = []
    
    for item in failed_list:
        cycle, scene = item
        if (cycle, scene) not in seen:
            seen.add((cycle, scene))
            unique_list.append(item)
    
    return unique_list

def main():
    """
    主函数
    """
    parser = argparse.ArgumentParser(description='解析物理仿真结果检查错误日志')
    parser.add_argument('--log_file', type=str, 
                       default=os.path.join(os.path.dirname(__file__), 'check_physics_utils.txt'),
                       help='错误日志文件路径')
    parser.add_argument('--remove_duplicates', action='store_true',
                       help='去除重复的循环-场景组合')
    parser.add_argument('--sort', action='store_true',
                       help='按循环和场景排序')
    
    args = parser.parse_args()
    
    print("🔍 开始解析错误日志...")
    print(f"📁 日志文件: {args.log_file}")
    print("=" * 60)
    
    # 解析日志文件
    failed_list = parse_error_log(args.log_file)
    
    if not failed_list:
        print("✅ 没有发现错误记录，所有检查都通过了！")
        return
    
    print(f"\n📊 解析完成，共发现 {len(failed_list)} 个错误记录")
    
    # 去除重复项（如果指定）
    if args.remove_duplicates:
        original_count = len(failed_list)
        failed_list = remove_duplicates(failed_list)
        duplicate_count = original_count - len(failed_list)
        print(f"🔄 去重完成，移除了 {duplicate_count} 个重复项，剩余 {len(failed_list)} 个唯一的失败组合")
    
    # 排序（如果指定）
    if args.sort:
        failed_list.sort(key=lambda x: (x[0], x[1]))  # 按循环号，然后按场景号排序
        print("📈 已按循环和场景排序")
    
    print("\n" + "=" * 60)
    print("🎯 失败的循环和场景列表:")
    print("=" * 60)
    
    # 输出列表
    print("failed_cycles_scenes = [")
    for i, (cycle, scene) in enumerate(failed_list):
        if i == len(failed_list) - 1:  # 最后一项不加逗号
            print(f"    [{cycle}, {scene}]")
        else:
            print(f"    [{cycle}, {scene}],")
    print("]")
    
    print(f"\n📋 总计: {len(failed_list)} 个失败的循环-场景组合")
    
    # 统计信息
    cycles = set(cycle for cycle, scene in failed_list)
    scenes = set(scene for cycle, scene in failed_list)
    
    print(f"🔄 涉及的循环数: {len(cycles)} 个 (循环: {sorted(cycles)})")
    print(f"🏷️ 涉及的场景数: {len(scenes)} 个 (场景: {sorted(scenes)})")
    
    # 按循环分组统计
    cycle_stats = {}
    for cycle, scene in failed_list:
        if cycle not in cycle_stats:
            cycle_stats[cycle] = []
        cycle_stats[cycle].append(scene)
    
    print("\n📈 按循环分组统计:")
    for cycle in sorted(cycle_stats.keys()):
        scenes_in_cycle = sorted(cycle_stats[cycle])
        print(f"  循环{cycle}: {len(scenes_in_cycle)} 个失败场景 {scenes_in_cycle}")

if __name__ == '__main__':
    main()
