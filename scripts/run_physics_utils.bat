@echo off
REM Windows 批处理脚本：运行 generate_dataset/physics_utils.py
REM 读取目录并执行 Python 脚本

REM 设置数据集目录
set DATA_DIR=G:/Diffusion_Suction_DataSet
REM 设置循环次数和场景数量
set CYCLE_NUM=100
set SCENE_NUM=50

REM 切换到项目根目录
cd /d %~dp0..

REM 执行 Python 脚本
"%~dp0..\diffusion_suction_conda\python.exe" generate_dataset\physics_utils.py --data_dir %DATA_DIR% --cycle_num %CYCLE_NUM% --scene_num %SCENE_NUM%

REM 可选：如需可视化，取消下行注释
REM "%~dp0..\diffusion_suction_conda\python.exe" generate_dataset\physics_utils.py --data_dir %DATA_DIR% --cycle_num %CYCLE_NUM% --scene_num %SCENE_NUM% --visualize
