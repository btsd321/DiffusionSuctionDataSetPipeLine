@echo off
echo Activating DiffusionSuctionDataSetPipeLine environment...
set PATH=D:\SoftWare\anaconda3\Scripts;D:\SoftWare\anaconda3;D:\SoftWare\anaconda3\Library\bin;%PATH%
call conda activate D:\Project\DiffusionSuctionDataSetPipeLine\diffusion_suction_conda
echo Environment activated successfully!
echo You can now use bpy, pybullet, opencv, and other packages.
