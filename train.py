"""
LipNet Training Script - Based on GRID Corpus Dataset
基于 GRID 语料库数据集的唇读训练脚本

使用方法:
    训练: python train.py --mode train --epochs 100
    测试: python train.py --mode test --checkpoint checkpoints/lipnet_best.keras
    推理: python train.py --mode inference --video path/to/video.mpg --checkpoint checkpoints/lipnet_best.keras
"""

import os
import sys
import glob
import string
import argparse
import json
from typing import List, Tuple, Optional
from datetime import datetime

# 配置 CUDA 库路径（在导入 TensorFlow 之前）
def setup_cuda_paths():
    """设置 CUDA 库路径"""
    try:
        import nvidia.cudnn
        cudnn_path = os.path.join(os.path.dirname(nvidia.cudnn.__file__), 'lib')
        if os.path.exists(cudnn_path):
            current_ld_path = os.environ.get('LD_LIBRARY_PATH', '')
            if cudnn_path not in current_ld_path:
                os.environ['LD_LIBRARY_PATH'] = f"{cudnn_path}:{current_ld_path}"
                print(f"Added cuDNN path: {cudnn_path}")
    except ImportError:
        pass

    try:
        import nvidia.cuda_runtime
        cuda_path = os.path.join(os.path.dirname(nvidia.cuda_runtime.__file__), 'lib')
        if os.path.exists(cuda_path):
            current_ld_path = os.environ.get('LD_LIBRARY_PATH', '')
            if cuda_path not in current_ld_path:
                os.environ['LD_LIBRARY_PATH'] = f"{cuda_path}:{current_ld_path}"
                print(f"Added CUDA runtime path: {cuda_path}")
    except ImportError:
        pass

setup_cuda_paths()

import numpy as np
import cv2
# 抑制 FFmpeg/libav 视频解码警告
os.environ["OPENCV_LOG_LEVEL"] = "SILENT"
cv2.setLogLevel(0)
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tqdm import tqdm

import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import (
    Conv3D, Dense, LSTM, Bidirectional, Dropout,
    MaxPool3D, Activation, Reshape, SpatialDropout3D,
    BatchNormalization, TimeDistributed, Flatten, Input
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import (
    LearningRateScheduler, ModelCheckpoint, EarlyStopping,
    TensorBoard, ReduceLROnPlateau, Callback
)

# 尝试导入 dlib，如果失败则使用简单的裁剪方法
try:
    import dlib
    DLIB_AVAILABLE = True
    print(f"dlib version: {dlib.__version__}")
except ImportError:
    DLIB_AVAILABLE = False
    print("WARNING: dlib not available, using simple mouth region extraction")

# ==================== 配置 ====================

class Config:
    """训练配置"""
    # 数据路径
    DATA_PATH = "./data"
    CHECKPOINT_DIR = "./checkpoints"
    LOG_DIR = "./logs"

    # 视频处理参数
    IMG_WIDTH = 140
    IMG_HEIGHT = 46
    MAX_VIDEO_LENGTH = 75
    MAX_LABEL_LENGTH = 40

    # 训练参数
    BATCH_SIZE = 8
    EPOCHS = 100
    LEARNING_RATE = 1e-4

    # 模型参数
    HIDDEN_DIM = 256
    DROPOUT_RATE = 0.5

    # 数据集分割
    TEST_SIZE = 0.2
    RANDOM_STATE = 42


# ==================== 词汇表 ====================

def create_vocabulary():
    """创建字符到数字的映射词汇表"""
    vocab = string.ascii_lowercase + "'?! "
    vocab = list(vocab)

    char_to_num = tf.keras.layers.StringLookup(vocabulary=vocab, oov_token="")
    num_to_char = tf.keras.layers.StringLookup(
        vocabulary=char_to_num.get_vocabulary(),
        oov_token="",
        invert=True
    )

    return char_to_num, num_to_char, vocab


# ==================== 唇部检测 ====================

class MouthDetector:
    """唇部区域检测器"""

    def __init__(self, predictor_path: str = "shape_predictor_68_face_landmarks.dat"):
        self.use_dlib = DLIB_AVAILABLE and os.path.exists(predictor_path)

        if self.use_dlib:
            self.detector = dlib.get_frontal_face_detector()
            self.predictor = dlib.shape_predictor(predictor_path)
            self.MOUTH_POINTS = list(range(48, 61))
            print("Using dlib for mouth detection")
        else:
            print("Using simple region extraction for mouth detection")

    def extract_mouth(self, frame: np.ndarray, padding: int = 30) -> Optional[np.ndarray]:
        """
        从帧中提取唇部区域

        Args:
            frame: BGR格式的视频帧
            padding: 唇部区域的填充像素

        Returns:
            唇部区域图像，如果检测失败返回None
        """
        if self.use_dlib:
            return self._extract_mouth_dlib(frame, padding)
        else:
            return self._extract_mouth_simple(frame)

    def _extract_mouth_dlib(self, frame: np.ndarray, padding: int = 30) -> Optional[np.ndarray]:
        """使用dlib提取唇部区域"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.detector(gray)

        if len(faces) == 0:
            return None

        face = faces[0]
        landmarks = self.predictor(gray, face)

        mouth_points = []
        for i in self.MOUTH_POINTS:
            x = landmarks.part(i).x
            y = landmarks.part(i).y
            mouth_points.append((x, y))

        x_coords = [pt[0] for pt in mouth_points]
        y_coords = [pt[1] for pt in mouth_points]
        min_x = min(x_coords)
        max_x = max(x_coords)
        min_y = min(y_coords)
        max_y = max(y_coords)

        # 添加填充
        min_x = max(0, min_x - padding)
        max_x = min(frame.shape[1], max_x + padding)
        min_y = max(0, min_y - padding)
        max_y = min(frame.shape[0], max_y + padding)

        mouth_region = frame[min_y:max_y, min_x:max_x]

        return mouth_region if mouth_region.size > 0 else None

    def _extract_mouth_simple(self, frame: np.ndarray) -> np.ndarray:
        """使用简单的区域裁剪提取唇部区域（不使用dlib）"""
        h, w = frame.shape[:2]

        # GRID数据集中，唇部通常位于帧的下半部分中央
        # 裁剪区域：垂直方向40%-70%，水平方向25%-75%
        y_start = int(h * 0.4)
        y_end = int(h * 0.7)
        x_start = int(w * 0.25)
        x_end = int(w * 0.75)

        mouth_region = frame[y_start:y_end, x_start:x_end]

        return mouth_region


# ==================== 数据加载 ====================

def load_video(path: str, mouth_detector: MouthDetector,
               img_width: int = 140, img_height: int = 46,
               max_frames: int = 75) -> tf.Tensor:
    """
    加载视频并提取唇部区域

    Args:
        path: 视频文件路径
        mouth_detector: 唇部检测器
        img_width: 输出宽度
        img_height: 输出高度
        max_frames: 最大帧数

    Returns:
        标准化后的视频帧张量
    """
    cap = cv2.VideoCapture(path)
    frames = []

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    for _ in range(frame_count):
        ret, frame = cap.read()
        if not ret:
            break

        # 提取唇部区域
        mouth_region = mouth_detector.extract_mouth(frame)

        if mouth_region is not None:
            # 调整大小
            mouth_region = cv2.resize(mouth_region, (img_width, img_height),
                                      interpolation=cv2.INTER_AREA)
            # 转换为灰度
            if len(mouth_region.shape) == 3:
                mouth_region = cv2.cvtColor(mouth_region, cv2.COLOR_BGR2GRAY)
            # 添加通道维度
            mouth_region = np.expand_dims(mouth_region, axis=-1)
            frames.append(mouth_region)

        # 限制最大帧数
        if len(frames) >= max_frames:
            break

    cap.release()

    if len(frames) == 0:
        # 返回空帧
        return tf.zeros((max_frames, img_height, img_width, 1), dtype=tf.float32)

    # 截断到最大帧数
    frames = frames[:max_frames]

    frames_tensor = tf.stack(frames)

    # 先转换为float32
    frames_tensor = tf.cast(frames_tensor, tf.float32)

    # 标准化
    mean = tf.reduce_mean(frames_tensor)
    std = tf.math.reduce_std(frames_tensor)

    # 避免除以零
    std = tf.maximum(std, 1e-6)

    return (frames_tensor - mean) / std


def load_alignment(path: str, char_to_num, max_label_length: int = 40) -> tf.Tensor:
    """
    加载对齐文件

    Args:
        path: 对齐文件路径
        char_to_num: 字符到数字的映射层
        max_label_length: 最大标签长度

    Returns:
        数字序列张量
    """
    with open(path, "r") as f:
        lines = f.readlines()

    tokens = []

    for line in lines:
        parts = line.strip().split()
        if len(parts) >= 3:
            text = parts[2]
            if text != 'sil':
                tokens.append(text)

    # 将单词连接成句子
    sentence = " ".join(tokens)
    chars = list(sentence)

    # 截断到最大长度
    chars = chars[:max_label_length]

    return char_to_num(chars)


def load_data(video_path: str, data_path: str,
              mouth_detector: MouthDetector, char_to_num,
              img_width: int = 140, img_height: int = 46) -> Tuple[tf.Tensor, tf.Tensor]:
    """
    加载单个样本的视频和标签

    Args:
        video_path: 视频文件路径
        data_path: 数据根目录
        mouth_detector: 唇部检测器
        char_to_num: 字符到数字的映射
        img_width: 图像宽度
        img_height: 图像高度

    Returns:
        (视频帧张量, 标签张量)
    """
    # 解析路径获取视频ID和说话人目录
    video_path_str = video_path.numpy().decode('UTF-8') if isinstance(video_path, tf.Tensor) else video_path
    video_path_str = video_path_str.replace("\\", "/")

    # 获取视频文件名（不含扩展名）
    video_id = os.path.splitext(os.path.basename(video_path_str))[0]

    # 获取说话人目录
    speaker_dir = os.path.dirname(video_path_str)

    # 构建对齐文件路径
    align_path = os.path.join(speaker_dir, "align", f"{video_id}.align")

    # 加载视频
    video_data = load_video(video_path_str, mouth_detector, img_width, img_height)

    # 加载标签
    char_num = load_alignment(align_path, char_to_num)

    return video_data, char_num


def create_tf_function(mouth_detector: MouthDetector, char_to_num,
                       img_width: int = 140, img_height: int = 46):
    """创建可映射的TensorFlow函数"""

    def mappable_function(path: tf.Tensor):
        result = tf.py_function(
            lambda p: load_data(p, "", mouth_detector, char_to_num, img_width, img_height),
            [path],
            (tf.float32, tf.int64)
        )
        return result

    return mappable_function


# ==================== 数据集创建 ====================

def get_all_videos(data_path: str, exclude_videos: List[str] = None) -> List[str]:
    """
    获取所有视频文件路径

    Args:
        data_path: 数据根目录
        exclude_videos: 需要排除的视频列表

    Returns:
        视频文件路径列表
    """
    exclude_videos = exclude_videos or []

    # 查找所有说话人目录
    speaker_dirs = sorted(glob.glob(os.path.join(data_path, "s*_processed")))

    all_videos = []

    for speaker_dir in speaker_dirs:
        # 获取该说话人的所有视频
        videos = glob.glob(os.path.join(speaker_dir, "*.mpg"))

        # 过滤排除的视频
        for video in videos:
            if video not in exclude_videos:
                # 检查是否有对应的对齐文件
                video_id = os.path.splitext(os.path.basename(video))[0]
                align_path = os.path.join(speaker_dir, "align", f"{video_id}.align")
                if os.path.exists(align_path):
                    all_videos.append(video)

    return all_videos


def create_dataset(video_paths: List[str],
                   mouth_detector: MouthDetector,
                   char_to_num,
                   batch_size: int = 8,
                   max_video_length: int = 75,
                   max_label_length: int = 40,
                   img_width: int = 140,
                   img_height: int = 46,
                   shuffle: bool = True,
                   cache: bool = True) -> tf.data.Dataset:
    """
    创建TensorFlow数据集

    Args:
        video_paths: 视频文件路径列表
        mouth_detector: 唇部检测器
        char_to_num: 字符到数字的映射
        batch_size: 批次大小
        max_video_length: 最大视频帧数
        max_label_length: 最大标签长度
        img_width: 图像宽度
        img_height: 图像高度
        shuffle: 是否打乱数据
        cache: 是否缓存数据

    Returns:
        TensorFlow数据集
    """

    def load_sample(path):
        path_str = path.numpy().decode('UTF-8')
        video_id = os.path.splitext(os.path.basename(path_str))[0]
        speaker_dir = os.path.dirname(path_str)
        align_path = os.path.join(speaker_dir, "align", f"{video_id}.align")

        video_data = load_video(path_str, mouth_detector, img_width, img_height, max_video_length)
        char_num = load_alignment(align_path, char_to_num, max_label_length)

        return video_data, char_num

    def mappable_function(path):
        result = tf.py_function(load_sample, [path], (tf.float32, tf.int64))
        return result

    # 创建数据集
    dataset = tf.data.Dataset.from_tensor_slices(video_paths)

    if shuffle:
        dataset = dataset.shuffle(min(500, len(video_paths)))

    dataset = dataset.map(mappable_function, num_parallel_calls=tf.data.AUTOTUNE)

    # 添加填充
    dataset = dataset.padded_batch(
        batch_size,
        padded_shapes=([max_video_length, img_height, img_width, 1], [max_label_length])
    )

    dataset = dataset.prefetch(tf.data.AUTOTUNE)

    if cache:
        dataset = dataset.cache()

    return dataset


# ==================== 损失函数 ====================

def CTCLoss(y_true, y_pred):
    """
    CTC损失函数

    Args:
        y_true: 真实标签
        y_pred: 预测值

    Returns:
        CTC损失值
    """
    batch_size = tf.cast(tf.shape(y_true)[0], tf.int64)
    input_len = tf.cast(tf.shape(y_pred)[1], tf.int64)
    label_len = tf.cast(tf.shape(y_true)[1], tf.int64)

    input_len = input_len * tf.ones(shape=(batch_size, 1), dtype=tf.int64)
    label_len = label_len * tf.ones(shape=(batch_size, 1), dtype=tf.int64)

    loss = tf.keras.backend.ctc_batch_cost(y_true, y_pred, input_len, label_len)

    return loss


# ==================== 模型定义 ====================

def create_lipnet_model(vocab_size: int,
                        input_shape: Tuple[int, int, int, int] = (75, 46, 140, 1),
                        hidden_dim: int = 256,
                        dropout_rate: float = 0.5) -> Model:
    """
    创建LipNet模型

    Args:
        vocab_size: 词汇表大小
        input_shape: 输入形状 (时间步, 高度, 宽度, 通道)
        hidden_dim: LSTM隐藏层维度
        dropout_rate: Dropout率

    Returns:
        Keras模型
    """
    model = Sequential([
        Input(shape=input_shape),

        # 第一个3D卷积块
        Conv3D(128, kernel_size=3, padding='same', activation='relu'),
        MaxPool3D(pool_size=(1, 2, 2)),

        # 第二个3D卷积块
        Conv3D(256, kernel_size=3, padding='same', activation='relu'),
        MaxPool3D(pool_size=(1, 2, 2)),

        # 第三个3D卷积块
        Conv3D(64, kernel_size=3, padding='same', activation='relu'),
        MaxPool3D(pool_size=(1, 2, 2)),

        # 重塑为序列
        Reshape([input_shape[0], -1]),

        # 双向LSTM层
        Bidirectional(LSTM(hidden_dim, return_sequences=True)),
        Dropout(dropout_rate),

        Bidirectional(LSTM(hidden_dim, return_sequences=True)),
        Dropout(dropout_rate),

        Bidirectional(LSTM(hidden_dim, return_sequences=True)),
        Dropout(dropout_rate),

        # 全连接层
        Dense(512, activation='relu', kernel_initializer='he_normal'),
        Dense(512, activation='relu', kernel_initializer='he_normal'),

        # 输出层
        Dense(vocab_size + 1, activation='softmax', kernel_initializer='he_normal')
    ])

    return model


# ==================== 回调函数 ====================

class ProduceExample(Callback):
    """在每个epoch结束时生成预测示例"""

    def __init__(self, dataset: tf.data.Dataset, num_to_char, name: str = ""):
        super().__init__()
        self.dataset = dataset
        self.num_to_char = num_to_char
        self.name = name
        self.iterator = None

    def on_epoch_end(self, epoch, logs=None):
        if self.iterator is None:
            self.iterator = iter(self.dataset)

        try:
            data = next(self.iterator)
        except StopIteration:
            self.iterator = iter(self.dataset)
            data = next(self.iterator)

        videos, labels = data

        # 确保批次大小足够
        if videos.shape[0] < 1:
            return

        # 获取预测
        yhat = self.model.predict(videos, verbose=0)

        # CTC解码
        decoded = tf.keras.backend.ctc_decode(
            yhat,
            [yhat.shape[1]] * videos.shape[0],
            greedy=True
        )[0][0].numpy()

        print(f"\n{'='*50}")
        print(f"Epoch {epoch + 1} - {self.name} Examples:")
        print('='*50)

        num_examples = min(3, len(decoded))
        for i in range(num_examples):
            # 原始标签
            original = tf.strings.reduce_join(
                self.num_to_char(labels[i])
            ).numpy().decode('utf-8').strip()

            # 预测结果
            prediction = tf.strings.reduce_join(
                self.num_to_char(decoded[i])
            ).numpy().decode('utf-8').strip()

            print(f"\nSample {i + 1}:")
            print(f"  Original:   '{original}'")
            print(f"  Prediction: '{prediction}'")

        print('='*50)


def scheduler(epoch: int, lr: float) -> float:
    """学习率调度器"""
    if epoch < 30:
        return lr
    elif epoch < 60:
        return lr * 0.5
    else:
        return lr * tf.math.exp(-0.1).numpy()


# ==================== 训练函数 ====================

def train(config: Config):
    """
    训练模型

    Args:
        config: 训练配置
    """
    print("\n" + "="*60)
    print("LipNet Training")
    print("="*60)

    # 创建目录
    os.makedirs(config.CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(config.LOG_DIR, exist_ok=True)

    # 创建词汇表
    char_to_num, num_to_char, vocab = create_vocabulary()
    vocab_size = char_to_num.vocabulary_size()
    print(f"Vocabulary size: {vocab_size}")

    # 创建唇部检测器
    mouth_detector = MouthDetector()

    # 获取所有视频
    print(f"\nLoading videos from: {config.DATA_PATH}")

    # 排除有问题的视频
    exclude_videos = [
        'lgal8n.mpg', 'bbaf4p.mpg', 'swwp3s.mpg',
        'lwik9s.mpg', 'pgwr6p.mpg'
    ]

    all_videos = get_all_videos(config.DATA_PATH)

    # 过滤排除的视频
    all_videos = [v for v in all_videos
                  if os.path.basename(v) not in exclude_videos]

    print(f"Found {len(all_videos)} videos")

    if len(all_videos) == 0:
        print("ERROR: No videos found!")
        return

    # 分割数据集
    train_videos, test_videos = train_test_split(
        all_videos,
        test_size=config.TEST_SIZE,
        random_state=config.RANDOM_STATE
    )

    print(f"Training videos: {len(train_videos)}")
    print(f"Test videos: {len(test_videos)}")

    # 创建数据集
    print("\nCreating datasets...")

    train_dataset = create_dataset(
        train_videos, mouth_detector, char_to_num,
        batch_size=config.BATCH_SIZE,
        max_video_length=config.MAX_VIDEO_LENGTH,
        max_label_length=config.MAX_LABEL_LENGTH,
        img_width=config.IMG_WIDTH,
        img_height=config.IMG_HEIGHT,
        shuffle=True,
        cache=False  # 禁用缓存避免部分读取问题
    )

    test_dataset = create_dataset(
        test_videos, mouth_detector, char_to_num,
        batch_size=config.BATCH_SIZE,
        max_video_length=config.MAX_VIDEO_LENGTH,
        max_label_length=config.MAX_LABEL_LENGTH,
        img_width=config.IMG_WIDTH,
        img_height=config.IMG_HEIGHT,
        shuffle=False,
        cache=False  # 禁用缓存避免部分读取问题
    )

    # 创建模型
    print("\nCreating model...")
    model = create_lipnet_model(
        vocab_size=vocab_size,
        input_shape=(config.MAX_VIDEO_LENGTH, config.IMG_HEIGHT, config.IMG_WIDTH, 1),
        hidden_dim=config.HIDDEN_DIM,
        dropout_rate=config.DROPOUT_RATE
    )

    model.summary()

    # 编译模型
    model.compile(
        optimizer=Adam(config.LEARNING_RATE),
        loss=CTCLoss
    )

    # 创建回调
    callbacks = [
        # 保存最佳模型（基于验证损失）
        ModelCheckpoint(
            os.path.join(config.CHECKPOINT_DIR, "lipnet_best.keras"),
            monitor="val_loss",
            save_best_only=True,
            verbose=1
        ),
        # 保存最佳模型（基于训练损失）
        ModelCheckpoint(
            os.path.join(config.CHECKPOINT_DIR, "lipnet_best_train.keras"),
            monitor="loss",
            save_best_only=True,
            verbose=1
        ),
        # 学习率调度
        LearningRateScheduler(scheduler),
        # 早停
        EarlyStopping(
            monitor="val_loss",
            patience=15,
            restore_best_weights=True,
            verbose=1
        ),
        # TensorBoard
        TensorBoard(
            log_dir=os.path.join(config.LOG_DIR, datetime.now().strftime("%Y%m%d-%H%M%S")),
            histogram_freq=1
        ),
        # 生成预测示例
        ProduceExample(test_dataset, num_to_char, "Validation"),
        ProduceExample(train_dataset, num_to_char, "Training"),
    ]

    # 训练
    print("\nStarting training...")
    history = model.fit(
        train_dataset,
        epochs=config.EPOCHS,
        validation_data=test_dataset,
        callbacks=callbacks,
        verbose=1
    )

    # 保存最终模型
    model.save(os.path.join(config.CHECKPOINT_DIR, "lipnet_final.keras"))
    print(f"\nModel saved to {config.CHECKPOINT_DIR}")

    # 保存训练历史
    history_path = os.path.join(config.CHECKPOINT_DIR, "training_history.json")
    with open(history_path, 'w') as f:
        json.dump({k: [float(v) for v in vals] for k, vals in history.history.items()}, f, indent=2)

    # 绘制训练曲线
    plot_training_history(history, config.CHECKPOINT_DIR)

    return model, history


def plot_training_history(history, save_dir: str):
    """绘制训练历史曲线"""
    plt.figure(figsize=(12, 4))

    # 损失曲线
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss Over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    # 学习率曲线（如果有）
    if 'lr' in history.history:
        plt.subplot(1, 2, 2)
        plt.plot(history.history['lr'], label='Learning Rate')
        plt.title('Learning Rate Over Epochs')
        plt.xlabel('Epoch')
        plt.ylabel('Learning Rate')
        plt.legend()
        plt.grid(True)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "training_history.png"), dpi=150)
    plt.show()
    print(f"Training history plot saved to {save_dir}/training_history.png")


# ==================== 测试函数 ====================

def test(config: Config, checkpoint_path: str):
    """
    测试模型

    Args:
        config: 配置
        checkpoint_path: 模型检查点路径
    """
    print("\n" + "="*60)
    print("LipNet Testing")
    print("="*60)

    # 创建词汇表
    char_to_num, num_to_char, vocab = create_vocabulary()
    vocab_size = char_to_num.vocabulary_size()

    # 创建唇部检测器
    mouth_detector = MouthDetector()

    # 获取测试视频
    all_videos = get_all_videos(config.DATA_PATH)
    _, test_videos = train_test_split(
        all_videos,
        test_size=config.TEST_SIZE,
        random_state=config.RANDOM_STATE
    )

    print(f"Test videos: {len(test_videos)}")

    # 创建测试数据集
    test_dataset = create_dataset(
        test_videos, mouth_detector, char_to_num,
        batch_size=config.BATCH_SIZE,
        max_video_length=config.MAX_VIDEO_LENGTH,
        max_label_length=config.MAX_LABEL_LENGTH,
        img_width=config.IMG_WIDTH,
        img_height=config.IMG_HEIGHT,
        shuffle=False,
        cache=False
    )

    # 加载模型
    print(f"\nLoading model from: {checkpoint_path}")
    model = tf.keras.models.load_model(
        checkpoint_path,
        custom_objects={'CTCLoss': CTCLoss}
    )

    # 评估
    print("\nEvaluating model...")

    total_samples = 0
    total_cer = 0.0
    total_wer = 0.0
    correct_predictions = 0

    results = []

    for batch_idx, (videos, labels) in enumerate(tqdm(test_dataset)):
        # 获取预测
        yhat = model.predict(videos, verbose=0)

        # CTC解码
        decoded = tf.keras.backend.ctc_decode(
            yhat,
            [yhat.shape[1]] * videos.shape[0],
            greedy=True
        )[0][0].numpy()

        for i in range(videos.shape[0]):
            # 原始标签
            original = tf.strings.reduce_join(
                num_to_char(labels[i])
            ).numpy().decode('utf-8').strip()

            # 预测结果
            prediction = tf.strings.reduce_join(
                num_to_char(decoded[i])
            ).numpy().decode('utf-8').strip()

            # 计算CER
            cer = calculate_cer(prediction, original)
            wer = calculate_wer(prediction, original)

            total_cer += cer
            total_wer += wer
            total_samples += 1

            if prediction == original:
                correct_predictions += 1

            results.append({
                'original': original,
                'prediction': prediction,
                'cer': cer,
                'wer': wer
            })

            # 打印前20个样本
            if total_samples <= 20:
                print(f"\nSample {total_samples}:")
                print(f"  Original:   '{original}'")
                print(f"  Prediction: '{prediction}'")
                print(f"  CER: {cer*100:.2f}%, WER: {wer*100:.2f}%")

    # 计算平均指标
    avg_cer = total_cer / max(total_samples, 1)
    avg_wer = total_wer / max(total_samples, 1)
    accuracy = correct_predictions / max(total_samples, 1)

    print("\n" + "="*60)
    print("EVALUATION RESULTS")
    print("="*60)
    print(f"Total Samples: {total_samples}")
    print(f"Average CER: {avg_cer*100:.2f}%")
    print(f"Average WER: {avg_wer*100:.2f}%")
    print(f"Exact Match Accuracy: {accuracy*100:.2f}% ({correct_predictions}/{total_samples})")
    print("="*60)

    # 保存结果
    results_path = os.path.join(config.CHECKPOINT_DIR, "test_results.json")
    with open(results_path, 'w') as f:
        json.dump({
            'total_samples': total_samples,
            'avg_cer': avg_cer,
            'avg_wer': avg_wer,
            'accuracy': accuracy,
            'correct_predictions': correct_predictions,
            'samples': results[:100]  # 保存前100个样本
        }, f, indent=2)

    print(f"\nResults saved to {results_path}")


def calculate_cer(prediction: str, target: str) -> float:
    """计算字符错误率 (CER)"""
    if len(target) == 0:
        return 1.0 if len(prediction) > 0 else 0.0

    # 编辑距离
    m, n = len(prediction), len(target)
    dp = [[0] * (n + 1) for _ in range(m + 1)]

    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if prediction[i-1] == target[j-1]:
                dp[i][j] = dp[i-1][j-1]
            else:
                dp[i][j] = min(dp[i-1][j], dp[i][j-1], dp[i-1][j-1]) + 1

    return dp[m][n] / len(target)


def calculate_wer(prediction: str, target: str) -> float:
    """计算词错误率 (WER)"""
    pred_words = prediction.split()
    target_words = target.split()

    if len(target_words) == 0:
        return 1.0 if len(pred_words) > 0 else 0.0

    # 词级别的编辑距离
    m, n = len(pred_words), len(target_words)
    dp = [[0] * (n + 1) for _ in range(m + 1)]

    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if pred_words[i-1] == target_words[j-1]:
                dp[i][j] = dp[i-1][j-1]
            else:
                dp[i][j] = min(dp[i-1][j], dp[i][j-1], dp[i-1][j-1]) + 1

    return dp[m][n] / len(target_words)


# ==================== 推理函数 ====================

def inference(config: Config, checkpoint_path: str, video_path: str):
    """
    对单个视频进行推理

    Args:
        config: 配置
        checkpoint_path: 模型检查点路径
        video_path: 视频文件路径
    """
    print("\n" + "="*60)
    print("LipNet Inference")
    print("="*60)

    # 创建词汇表
    char_to_num, num_to_char, vocab = create_vocabulary()

    # 创建唇部检测器
    mouth_detector = MouthDetector()

    # 加载模型
    print(f"Loading model from: {checkpoint_path}")
    model = tf.keras.models.load_model(
        checkpoint_path,
        custom_objects={'CTCLoss': CTCLoss}
    )

    # 加载视频
    print(f"Processing video: {video_path}")
    video_data = load_video(video_path, mouth_detector, config.IMG_WIDTH, config.IMG_HEIGHT)

    # 填充到最大长度
    if video_data.shape[0] < config.MAX_VIDEO_LENGTH:
        padding = tf.zeros((
            config.MAX_VIDEO_LENGTH - video_data.shape[0],
            config.IMG_HEIGHT, config.IMG_WIDTH, 1
        ))
        video_data = tf.concat([video_data, padding], axis=0)
    else:
        video_data = video_data[:config.MAX_VIDEO_LENGTH]

    # 添加批次维度
    video_data = tf.expand_dims(video_data, axis=0)

    # 预测
    yhat = model.predict(video_data, verbose=0)

    # CTC解码
    decoded = tf.keras.backend.ctc_decode(
        yhat,
        [config.MAX_VIDEO_LENGTH],
        greedy=True
    )[0][0].numpy()

    # 转换为文本
    prediction = tf.strings.reduce_join(
        num_to_char(decoded[0])
    ).numpy().decode('utf-8').strip()

    print("\n" + "="*60)
    print(f"Video: {video_path}")
    print(f"Prediction: '{prediction}'")
    print("="*60)

    # 如果有对应的标签文件，显示真实标签
    video_id = os.path.splitext(os.path.basename(video_path))[0]
    align_path = os.path.join(os.path.dirname(video_path), "align", f"{video_id}.align")

    if os.path.exists(align_path):
        label = load_alignment(align_path, char_to_num)
        original = tf.strings.reduce_join(
            num_to_char(label)
        ).numpy().decode('utf-8').strip()

        print(f"Ground Truth: '{original}'")
        print(f"CER: {calculate_cer(prediction, original)*100:.2f}%")
        print(f"WER: {calculate_wer(prediction, original)*100:.2f}%")

    return prediction


# ==================== 主函数 ====================

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description='LipNet: End-to-End Sentence-level Lipreading',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument('--mode', type=str, default='train',
                       choices=['train', 'test', 'inference'],
                       help='运行模式')

    parser.add_argument('--data_path', type=str, default='./data',
                       help='数据集路径')

    parser.add_argument('--checkpoint', type=str, default=None,
                       help='模型检查点路径')

    parser.add_argument('--video', type=str, default=None,
                       help='推理视频路径')

    parser.add_argument('--epochs', type=int, default=100,
                       help='训练轮数')

    parser.add_argument('--batch_size', type=int, default=8,
                       help='批次大小')

    parser.add_argument('--learning_rate', type=float, default=1e-4,
                       help='学习率')

    return parser.parse_args()


def main():
    """主函数"""
    args = parse_args()

    # 创建配置
    config = Config()
    config.DATA_PATH = args.data_path
    config.EPOCHS = args.epochs
    config.BATCH_SIZE = args.batch_size
    config.LEARNING_RATE = args.learning_rate

    # 设置GPU内存增长
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print(f"Found {len(gpus)} GPU(s)")
        except RuntimeError as e:
            print(f"GPU setup error: {e}")
    else:
        print("No GPU found, using CPU")

    if args.mode == 'train':
        train(config)

    elif args.mode == 'test':
        if args.checkpoint is None:
            print("ERROR: --checkpoint is required for test mode")
            return
        test(config, args.checkpoint)

    elif args.mode == 'inference':
        if args.checkpoint is None:
            print("ERROR: --checkpoint is required for inference mode")
            return
        if args.video is None:
            print("ERROR: --video is required for inference mode")
            return
        inference(config, args.checkpoint, args.video)


if __name__ == "__main__":
    main()
