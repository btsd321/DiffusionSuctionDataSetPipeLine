#!/bin/bash

# 直接使用脚本的绝对路径来确定工作空间
SCRIPT_PATH="$(readlink -f "${BASH_SOURCE[0]}")"
SCRIPT_DIR="$(dirname "$SCRIPT_PATH")"
WORK_SPACE_DIR="$(dirname "$SCRIPT_DIR")"
echo "Work Space is :$WORK_SPACE_DIR"

cd $WORK_SPACE_DIR

# 环境配置
# 检查conda是否已初始化，如果没有则初始化
if ! command -v conda &> /dev/null; then
    source "$HOME/SoftWare/miniconda3/etc/profile.d/conda.sh"
fi
conda activate linux_conda
export CUDA_HOME=/usr/local/cuda-11.8
CUDA_LIB="$CUDA_HOME/lib64"
CUDA_CUPTI="$CUDA_HOME/extras/CUPTI/lib64"
CUDA_BIN="$CUDA_HOME/bin"
# 只在LD_LIBRARY_PATH中未包含时再添加
if [[ ":$LD_LIBRARY_PATH:" != *":$CUDA_LIB:"* ]]; then
    export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:$CUDA_LIB"
fi
if [[ ":$LD_LIBRARY_PATH:" != *":$CUDA_CUPTI:"* ]]; then
    export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:$CUDA_CUPTI"
fi

# 只在PATH中未包含时再添加
if [[ ":$PATH:" != *":$CUDA_BIN:"* ]]; then
    export PATH="$PATH:$CUDA_BIN"
fi

# GPU配置
export CUDA_VISIBLE_DEVICES=0

# 执行渲染工具
# --data_dir: 数据集根目录路径
# --cycle_list: 循环序列范围 [起始, 结束]
# --scene_list: 场景序列范围 [起始, 结束]
# --use_gpu: 启用GPU渲染加速
python "$WORK_SPACE_DIR/generate_dataset/render_utils.py" \
    --data_dir "/home/lixinlong/Data/Diffusion_Suction_DataSet" \
    --cycle_list "[75,99]" \
    --scene_list "[1,50]" \
    --use_gpu
