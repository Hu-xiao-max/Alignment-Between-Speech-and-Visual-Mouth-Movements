# Alignment-Between-Speech-and-Visual-Mouth-Movements
# V1
1.Dataset download

https://www.kaggle.com/datasets/jedidiahangekouakou/grid-corpus-dataset-for-training-lipnet/data

unzip in follow path：/home/alien/Code/Alignment-Between-Speech-and-Visual-Mouth-Movements/data/

2.python3 main.py

  Run with various modes:
  ## Training
  python main.py --mode train --data_path ./data --epochs 50

  ## Testing
  python main.py --mode test --checkpoint checkpoints/lipnet_best.pth

  ## Inference
  python main.py --mode inference --checkpoint checkpoints/lipnet_best.pth --video video.mp4

  ## with improvement
  python main.py --mode train --epochs 100 --blank_penalty 0.3 --batch_size 16


# V2

## 安装依赖
  pip install -r requirements.txt

  ### 下载 dlib 人脸特征点检测模型（optional）
  wget https://github.com/italojs/facial-landmarks-recognition/raw/master/shape_predictor_68_face_landmarks.dat

  ### 训练模型
  ./run_train.sh --mode train --epochs 100 --batch_size 8

  ### 测试模型
  ./run_train.sh  python train.py --mode test --checkpoint checkpoints/lipnet_best.keras

  ### 单视频推理
  ./run_train.sh python train.py --mode inference --checkpoint checkpoints/lipnet_best.keras --video ./data/s1_processed/bbaf2n.mpg
