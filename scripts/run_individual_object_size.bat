@echo off
REM Windows 批处理脚本：运行 generate_dataset/individual_object_size.py
REM 读取目录并执行 Python 脚本

REM 设置数据集目录
set DATA_DIR=D:/Project/DiffusionSuctionDataSetPipeLine/Data/Diffusion_Suction_DataSet
REM 设置循环次数和场景数量
set CYCLE_LIST="0"
set SCENE_LIST="[1,5]"

REM 切换到项目根目录
cd /d %~dp0..

REM 执行 Python 脚本
"%~dp0../.conda_env/python.exe" generate_dataset/individual_object_size.py ^
    --data_dir %DATA_DIR% ^
    --cycle_list %CYCLE_LIST% ^
    --scene_list %SCENE_LIST% ^
