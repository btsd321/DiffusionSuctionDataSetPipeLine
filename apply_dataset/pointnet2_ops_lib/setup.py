import glob
import os
import os.path as osp

from setuptools import find_packages, setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

# 获取当前文件所在目录
this_dir = osp.dirname(osp.abspath(__file__))

# 指定C++/CUDA扩展源码目录
_ext_src_root = osp.join("pointnet2_ops", "_ext-src")
# 搜索所有.cpp和.cu源文件
_ext_sources = glob.glob(osp.join(_ext_src_root, "src", "*.cpp")) + glob.glob(
    osp.join(_ext_src_root, "src", "*.cu")
)
# 搜索所有头文件
_ext_headers = glob.glob(osp.join(_ext_src_root, "include", "*"))

# 指定依赖的PyTorch版本
requirements = ["torch>=1.4"]

# 读取版本号(从pointnet2_ops/_version.py文件中)
exec(open(osp.join("pointnet2_ops", "_version.py")).read())

# 指定支持的CUDA架构(可根据实际显卡情况调整)
os.environ["TORCH_CUDA_ARCH_LIST"] = "3.7+PTX;5.0;6.0;6.1;6.2;7.0;7.5"

# 配置并编译pointnet2_ops模块
setup(
    name="pointnet2_ops",  # 包名
    version=__version__,   # 版本号
    author="Erik Wijmans", # 作者
    packages=find_packages(),  # 自动查找所有子包
    install_requires=requirements,  # 安装依赖
    ext_modules=[
        CUDAExtension(
            name="pointnet2_ops._ext",  # 扩展模块名
            sources=_ext_sources,       # 源文件列表
            extra_compile_args={
                "cxx": ["-O3"],  # C++编译优化选项
                "nvcc": ["-O3", "-Xfatbin", "-compress-all"],  # CUDA编译优化选项
            },
            include_dirs=[osp.join(this_dir, _ext_src_root, "include")],  # 头文件目录
        )
    ],
    cmdclass={"build_ext": BuildExtension},  # 使用PyTorch的BuildExtension进行编译
    include_package_data=True,  # 包含包内的所有数据文件
)
