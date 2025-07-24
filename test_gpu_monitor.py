#!/usr/bin/env python3
"""
简单的GPU监控测试脚本
"""
import subprocess
import time
import os

def test_gpu_monitoring():
    """测试GPU监控功能"""
    print("GPU监控测试开始...")
    
    # 检查可用GPU
    try:
        result = subprocess.run(['nvidia-smi', '--list-gpus'], 
                              capture_output=True, text=True, check=True)
        print("可用GPU:")
        print(result.stdout)
    except Exception as e:
        print(f"无法检测GPU: {e}")
        return
    
    # 设置要测试的GPU ID
    gpu_ids = os.environ.get('CUDA_VISIBLE_DEVICES', '0,1,2,3').split(',')
    gpu_ids = [int(gid.strip()) for gid in gpu_ids if gid.strip()]
    
    print(f"测试GPU ID: {gpu_ids}")
    
    # 对每个GPU进行基础监控测试
    for gpu_id in gpu_ids:
        print(f"\n=== 测试GPU {gpu_id} ===")
        
        try:
            # 基础信息
            result = subprocess.run([
                'nvidia-smi', '-i', str(gpu_id),
                '--query-gpu=name,memory.total,memory.used,utilization.gpu,temperature.gpu',
                '--format=csv,noheader'
            ], capture_output=True, text=True, check=True)
            
            print(f"GPU {gpu_id} 基础信息: {result.stdout.strip()}")
            
            # 检查计算进程
            proc_result = subprocess.run([
                'nvidia-smi', '-i', str(gpu_id), 
                '--query-compute-apps=pid,process_name,used_memory',
                '--format=csv,noheader'
            ], capture_output=True, text=True)
            
            if proc_result.stdout.strip():
                print(f"GPU {gpu_id} 计算进程:")
                print(proc_result.stdout.strip())
            else:
                print(f"GPU {gpu_id} 无活动计算进程")
                
        except Exception as e:
            print(f"GPU {gpu_id} 监控失败: {e}")
    
    print("\n=== 实时监控测试 (10秒) ===")
    
    # 实时监控第一个GPU
    test_gpu = gpu_ids[0]
    for i in range(5):
        try:
            result = subprocess.run([
                'nvidia-smi', '-i', str(test_gpu),
                '--query-gpu=utilization.gpu,memory.used,power.draw',
                '--format=csv,noheader,nounits'
            ], capture_output=True, text=True, check=True)
            
            util, mem, power = result.stdout.strip().split(', ')
            print(f"[{i+1}/5] GPU {test_gpu}: 利用率={util:>3s}%, 显存={mem:>6s}MB, 功耗={power:>6s}W")
            
        except Exception as e:
            print(f"监控错误: {e}")
        
        time.sleep(2)

if __name__ == "__main__":
    test_gpu_monitoring()
