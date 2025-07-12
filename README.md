DiffusionSuction DataSet Pipeline

出自论文Diffusion Suction Grasping with Large-Scale Parcel Dataset(arXiv:2502.07238v2 [cs.CV] 17 Mar 2025)

我这边并非原作者只是将其数据集生成脚本copy了一份使用，主要知识产权属于原作者，我这边仅作部分更改，商用请联系原作者

使用步骤：
1.下载原作者数据集https://drive.google.com/drive/folders/1l4jz7LE7HXdn2evylodggReTTnip7J1Q中的BOX和OBJ文件夹放到自己电脑的某个目录下，磁盘一定要预留足够的空间，推荐预留1T

2.根据environment.yml安装合适的conda环境

3.运行generate_dataset下的physics_utils.py文件，推荐附带以下参数：[
    "--data_dir", "G:/Diffusion_Suction_DataSet",
    "--cycle_num", "100",
    "--scene_num", "50"
    // "--visualize"
]