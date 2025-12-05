#!/bin/bash
# LipNet training startup script
# Automatically configure CUDA environment and start training

# Activate conda environment
source ~/anaconda3/etc/profile.d/conda.sh
conda activate pyrep

# Set cuDNN library path
CUDNN_PATH=$(python -c "import nvidia.cudnn; import os; print(os.path.join(os.path.dirname(nvidia.cudnn.__file__), 'lib'))")
export LD_LIBRARY_PATH=$CUDNN_PATH:$LD_LIBRARY_PATH

echo "=============================================="
echo "CUDA Environment Setup Complete"
echo "cuDNN path: $CUDNN_PATH"
echo "=============================================="

# Check GPU
python -c "import tensorflow as tf; gpus=tf.config.list_physical_devices('GPU'); print(f'Found {len(gpus)} GPU(s): {gpus}')"

# Run training script, pass all arguments
python train.py "$@"   
