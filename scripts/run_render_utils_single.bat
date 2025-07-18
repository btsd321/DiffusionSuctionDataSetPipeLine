@echo off
REM Windows 批处理脚本：运行 generate_dataset/render_utils_single.py
REM 读取目录并执行 Python 脚本

REM 设置数据集目录
set DATA_DIR=G:/Diffusion_Suction_DataSet
REM 设置循环次数和场景数量
set CYCLE_LIST="[0,10]"
set SCENE_LIST="[34,50]"

REM 切换到项目根目录
cd /d %~dp0..

REM 执行 Python 脚本
"%~dp0..\diffusion_suction_conda\python.exe" generate_dataset\render_utils_single.py ^
    --data_dir %DATA_DIR% ^
    --cycle_list %CYCLE_LIST% ^
    --scene_list %SCENE_LIST% ^
    --use_gpu
