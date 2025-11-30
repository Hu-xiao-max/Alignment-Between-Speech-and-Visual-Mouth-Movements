#!/bin/bash
#SBATCH -p gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --job-name=misalign_train
#SBATCH --output=logs/misalign_train_%j.log
#SBATCH --error=logs/misalign_train_%j.err


module purge

module load cuda/12.1.1

eval "$(conda shell.bash hook)"

# Activate your environment
conda activate prj2

# Verify GPU
nvidia-smi
python -c "import torch; print('CUDA available:', torch.cuda.is_available()); print('Device count:', torch.cuda.device_count())"

# Change to project directory
cd /home/gu.yunp/Alignment-Between-Speech-and-Visual-Mouth-Movements

# Create logs directory if not exists
mkdir -p logs

# Start training
python misalignment_detection_train.py \
  --data_path ./data \
  --checkpoint lipnet_final.pth \
  --detector_checkpoint misalignment_detector.pth \
  --max_samples 3300 \
  --epochs 20 \
  --max_shift_frames 20 \
  --hidden_dim 512 \
  --batch_size 64 \
  --save_every 5 \
  --log_dir logs

