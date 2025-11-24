#!/bin/bash
# LipNet 训练启动脚本
# 自动配置 CUDA 环境并启动训练

# 激活 conda 环境
source ~/anaconda3/etc/profile.d/conda.sh
conda activate pyrep

# 设置 cuDNN 库路径
CUDNN_PATH=$(python -c "import nvidia.cudnn; import os; print(os.path.join(os.path.dirname(nvidia.cudnn.__file__), 'lib'))")
export LD_LIBRARY_PATH=$CUDNN_PATH:$LD_LIBRARY_PATH

echo "=============================================="
echo "CUDA Environment Setup Complete"
echo "cuDNN path: $CUDNN_PATH"
echo "=============================================="

# 检查 GPU
python -c "import tensorflow as tf; gpus=tf.config.list_physical_devices('GPU'); print(f'Found {len(gpus)} GPU(s): {gpus}')"

# 运行训练脚本，传递所有参数
python train.py "$@"
