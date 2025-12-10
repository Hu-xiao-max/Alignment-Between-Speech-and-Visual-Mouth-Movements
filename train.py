# Xiao Hu, Yibo Wang, Xinran Tao, Yunpei Gu
# 2025-11-30
# CS 7180 Advanced Perception

"""
LipNet Training Script - Based on GRID Corpus Dataset

description:
    train: python train.py --mode train --epochs 100
    test: python train.py --mode test --checkpoint checkpoints/lipnet_best.keras
    inference: python train.py --mode inference --video path/to/video.mpg --checkpoint checkpoints/lipnet_best.keras
"""

import os
import sys
import glob
import string
import argparse
import json
from typing import List, Tuple, Optional
from datetime import datetime

# configure CUDA library paths (before importing TensorFlow)
def setup_cuda_paths():
    """set CUDA library paths"""
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
# Suppress FFmpeg/libav video decoding warning
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

# try to import dlib, if fails, use simple mouth region extraction
try:
    import dlib
    DLIB_AVAILABLE = True
    print(f"dlib version: {dlib.__version__}")
except ImportError:
    DLIB_AVAILABLE = False
    print("WARNING: dlib not available, using simple mouth region extraction")

# ==================== configuration ====================

class Config:
    """training configuration"""
    # data path
    DATA_PATH = "./data"
    CHECKPOINT_DIR = "./checkpoints"
    LOG_DIR = "./logs"

    # video processing parameters
    IMG_WIDTH = 140
    IMG_HEIGHT = 46
    MAX_VIDEO_LENGTH = 75
    MAX_LABEL_LENGTH = 40

    # training parameters
    BATCH_SIZE = 8
    EPOCHS = 100
    LEARNING_RATE = 1e-4

    # model parameters
    HIDDEN_DIM = 256
    DROPOUT_RATE = 0.5

    # dataset split
    TEST_SIZE = 0.2
    RANDOM_STATE = 42


# ==================== vocabulary ====================

def create_vocabulary():
    """create character to number mapping vocabulary"""
    vocab = string.ascii_lowercase + "'?! "
    vocab = list(vocab)

    char_to_num = tf.keras.layers.StringLookup(vocabulary=vocab, oov_token="")
    num_to_char = tf.keras.layers.StringLookup(
        vocabulary=char_to_num.get_vocabulary(),
        oov_token="",
        invert=True
    )

    return char_to_num, num_to_char, vocab


# ==================== mouth detection ====================

class MouthDetector:
    """mouth region detector"""

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
        Extract the mouth region from the frame

        Args:
            frame: BGR format video frame
            padding: Padding pixels for the mouth region

        Returns:
            Mouth region image, None if detection fails
        """
        if self.use_dlib:
            return self._extract_mouth_dlib(frame, padding)
        else:
            return self._extract_mouth_simple(frame)

    def _extract_mouth_dlib(self, frame: np.ndarray, padding: int = 30) -> Optional[np.ndarray]:
        """Extract the mouth region using dlib"""
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

        # add padding
        min_x = max(0, min_x - padding)
        max_x = min(frame.shape[1], max_x + padding)
        min_y = max(0, min_y - padding)
        max_y = min(frame.shape[0], max_y + padding)

        mouth_region = frame[min_y:max_y, min_x:max_x]

        return mouth_region if mouth_region.size > 0 else None

    def _extract_mouth_simple(self, frame: np.ndarray) -> np.ndarray:
        """Extract the mouth region using simple region cropping (without dlib)"""
        h, w = frame.shape[:2]

        # GRID data, mouth usually located in the lower half of the frame
        # cropping region: vertical 40%-70%, horizontal 25%-75%
        y_start = int(h * 0.4)
        y_end = int(h * 0.7)
        x_start = int(w * 0.25)
        x_end = int(w * 0.75)

        mouth_region = frame[y_start:y_end, x_start:x_end]

        return mouth_region


# ==================== data loading ====================

def load_video(path: str, mouth_detector: MouthDetector,
               img_width: int = 140, img_height: int = 46,
               max_frames: int = 75) -> tf.Tensor:
    """
    Load video and extract mouth region

    Args:
        path: video file path
        mouth_detector: mouth detector
        img_width: output width
        img_height: output height
        max_frames: maximum number of frames

    Returns:
        normalized video frames tensor
    """
    cap = cv2.VideoCapture(path)
    frames = []

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    for _ in range(frame_count):
        ret, frame = cap.read()
        if not ret:
            break

        # extract mouth region
        mouth_region = mouth_detector.extract_mouth(frame)

        if mouth_region is not None:
            # resize
            mouth_region = cv2.resize(mouth_region, (img_width, img_height),
                                      interpolation=cv2.INTER_AREA)
            # convert to grayscale
            if len(mouth_region.shape) == 3:
                mouth_region = cv2.cvtColor(mouth_region, cv2.COLOR_BGR2GRAY)
            # add channel dimension
            mouth_region = np.expand_dims(mouth_region, axis=-1)
            frames.append(mouth_region)

        # limit maximum number of frames
        if len(frames) >= max_frames:
            break

    cap.release()

    if len(frames) == 0:
        # return empty frames
        return tf.zeros((max_frames, img_height, img_width, 1), dtype=tf.float32)

    # truncate to maximum number of frames
    frames = frames[:max_frames]

    frames_tensor = tf.stack(frames)

    # convert to float32
    frames_tensor = tf.cast(frames_tensor, tf.float32)

    # standardize
    mean = tf.reduce_mean(frames_tensor)
    std = tf.math.reduce_std(frames_tensor)

    # avoid division to zero
    std = tf.maximum(std, 1e-6)

    return (frames_tensor - mean) / std


def load_alignment(path: str, char_to_num, max_label_length: int = 40) -> tf.Tensor:
    """
    Load alignment file

    Args:
        path: alignment file path
        char_to_num: character to number mapping layer
        max_label_length: maximum label length

    Returns:
        number sequence tensor
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

    # Concatenate words into a sentence
    sentence = " ".join(tokens)
    chars = list(sentence)

    # truncate to maximum length
    chars = chars[:max_label_length]

    return char_to_num(chars)


def load_data(video_path: str, data_path: str,
              mouth_detector: MouthDetector, char_to_num,
              img_width: int = 140, img_height: int = 46) -> Tuple[tf.Tensor, tf.Tensor]:
    """
    Load a single sample of video and label

    Args:
        video_path: path of the video file
        data_path: root directory of the data
        mouth_detector: mouth detector
        char_to_num: mapping of characters to numbers
        img_width: width of the image
        img_height: height of the image

    Returns:
        (tensor of video frames, tensor of labels)
    """
    # parse path to get video ID and speaker directory
    video_path_str = video_path.numpy().decode('UTF-8') if isinstance(video_path, tf.Tensor) else video_path
    video_path_str = video_path_str.replace("\\", "/")

    # get video file name (without extension)
    video_id = os.path.splitext(os.path.basename(video_path_str))[0]

    # get speaker directory
    speaker_dir = os.path.dirname(video_path_str)

    # build alignment file path
    align_path = os.path.join(speaker_dir, "align", f"{video_id}.align")

    # load video
    video_data = load_video(video_path_str, mouth_detector, img_width, img_height)

    # load label
    char_num = load_alignment(align_path, char_to_num)

    return video_data, char_num


def create_tf_function(mouth_detector: MouthDetector, char_to_num,
                       img_width: int = 140, img_height: int = 46):
    """Create a mappable TensorFlow function"""

    def mappable_function(path: tf.Tensor):
        result = tf.py_function(
            lambda p: load_data(p, "", mouth_detector, char_to_num, img_width, img_height),
            [path],
            (tf.float32, tf.int64)
        )
        return result

    return mappable_function


# ==================== dataset creation ====================

def get_all_videos(data_path: str, exclude_videos: List[str] = None) -> List[str]:
    """
    Get all video file paths

    Args:
        data_path: root directory of the data
        exclude_videos: list of videos to exclude

    Returns:
        list of video file paths
    """
    exclude_videos = exclude_videos or []

    # find all speaker directories
    speaker_dirs = sorted(glob.glob(os.path.join(data_path, "s*_processed")))

    all_videos = []

    for speaker_dir in speaker_dirs:
         # get all videos for this speaker
        videos = glob.glob(os.path.join(speaker_dir, "*.mpg"))

        # filter out excluded videos
        for video in videos:
            if video not in exclude_videos:
                # check if there is a corresponding alignment file
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
    Create TensorFlow dataset

    Args:
        video_paths: list of video file paths
        mouth_detector: lip detector
        char_to_num: character to number mapping
        batch_size: batch size
        max_video_length: maximum video length
        max_label_length: maximum label length
        img_width: image width
        img_height: image height
        shuffle: whether to shuffle data
        cache: whether to cache data

    Returns:
        TensorFlow dataset
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

    # Create the dataset
    dataset = tf.data.Dataset.from_tensor_slices(video_paths)

    if shuffle:
        dataset = dataset.shuffle(min(500, len(video_paths)))

    dataset = dataset.map(mappable_function, num_parallel_calls=tf.data.AUTOTUNE)

    # Add padding
    dataset = dataset.padded_batch(
        batch_size,
        padded_shapes=([max_video_length, img_height, img_width, 1], [max_label_length])
    )

    dataset = dataset.prefetch(tf.data.AUTOTUNE)

    if cache:
        dataset = dataset.cache()

    return dataset


# ==================== loss function ====================

def CTCLoss(y_true, y_pred):
    """
    CTC loss function

    Args:
        y_true: true labels
        y_pred: predicted values

    Returns:
        CTC loss value
    """
    batch_size = tf.cast(tf.shape(y_true)[0], tf.int64)
    input_len = tf.cast(tf.shape(y_pred)[1], tf.int64)
    
    # FIX：correct label length by counting non-zero values
    label_len = tf.math.count_nonzero(y_true, axis=1, dtype=tf.int64)

    input_len = input_len * tf.ones(shape=(batch_size, 1), dtype=tf.int64)
    label_len = tf.expand_dims(label_len, axis=1)

    loss = tf.keras.backend.ctc_batch_cost(y_true, y_pred, input_len, label_len)

    return loss


# ==================== model definition ====================

def create_lipnet_model(vocab_size: int,
                        input_shape: Tuple[int, int, int, int] = (75, 46, 140, 1),
                        hidden_dim: int = 256,
                        dropout_rate: float = 0.5) -> Model:
    """
    Create LipNet model

    Args:
        vocab_size: Size of the vocabulary
        input_shape: Input shape (time steps, height, width, channels)
        hidden_dim: LSTM hidden layer dimension
        dropout_rate: Dropout rate

    Returns:
        Keras model
    """
    model = Sequential([
        Input(shape=input_shape),

        # first 3D convolution block
        Conv3D(128, kernel_size=3, padding='same', activation='relu'),
        MaxPool3D(pool_size=(1, 2, 2)),

        # second 3D convolution block
        Conv3D(256, kernel_size=3, padding='same', activation='relu'),
        MaxPool3D(pool_size=(1, 2, 2)),

        # third 3D convolution block
        Conv3D(64, kernel_size=3, padding='same', activation='relu'),
        MaxPool3D(pool_size=(1, 2, 2)),

        # reshape to sequence
        Reshape([input_shape[0], -1]),

        # bidirectional LSTM layer
        Bidirectional(LSTM(hidden_dim, return_sequences=True)),
        Dropout(dropout_rate),

        Bidirectional(LSTM(hidden_dim, return_sequences=True)),
        Dropout(dropout_rate),

        Bidirectional(LSTM(hidden_dim, return_sequences=True)),
        Dropout(dropout_rate),

        # fully connected layer
        Dense(512, activation='relu', kernel_initializer='he_normal'),
        Dense(512, activation='relu', kernel_initializer='he_normal'),

        # output layer
        Dense(vocab_size + 1, activation='softmax', kernel_initializer='he_normal')
    ])

    return model


# ==================== callbacks ====================

class ProduceExample(Callback):
    """Generate prediction examples at the end of each epoch"""

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

        # Ensure batch size is sufficient
        if videos.shape[0] < 1:
            return

        # Get predictions
        yhat = self.model.predict(videos, verbose=0)

        # CTC decoding
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
            # original label
            original = tf.strings.reduce_join(
                self.num_to_char(labels[i])
            ).numpy().decode('utf-8').strip()

            # prediction result
            prediction = tf.strings.reduce_join(
                self.num_to_char(decoded[i])
            ).numpy().decode('utf-8').strip()

            print(f"\nSample {i + 1}:")
            print(f"  Original:   '{original}'")
            print(f"  Prediction: '{prediction}'")

        print('='*50)


def scheduler(epoch: int, lr: float) -> float:
    """Learning rate scheduler"""
    if epoch < 30:
        return lr
    elif epoch < 60:
        return lr * 0.5
    else:
        return lr * tf.math.exp(-0.1).numpy()


# ==================== training function ====================

def train(config: Config):
    """
    Train model

    Args:
        config: training configuration
    """
    print("\n" + "="*60)
    print("LipNet Training")
    print("="*60)

    # Create directories
    os.makedirs(config.CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(config.LOG_DIR, exist_ok=True)

    # Create vocabulary
    char_to_num, num_to_char, vocab = create_vocabulary()
    vocab_size = char_to_num.vocabulary_size()
    print(f"Vocabulary size: {vocab_size}")

    # Create mouth detector
    mouth_detector = MouthDetector()

    # Loading videos
    print(f"\nLoading videos from: {config.DATA_PATH}")

    # Filter out excluded videos
    exclude_videos = [
        'lgal8n.mpg', 'bbaf4p.mpg', 'swwp3s.mpg',
        'lwik9s.mpg', 'pgwr6p.mpg'
    ]

    all_videos = get_all_videos(config.DATA_PATH)

    # Filter out excluded videos
    all_videos = [v for v in all_videos
                  if os.path.basename(v) not in exclude_videos]

    print(f"Found {len(all_videos)} videos")

    if len(all_videos) == 0:
        print("ERROR: No videos found!")
        return

    # Split dataset
    train_videos, test_videos = train_test_split(
        all_videos,
        test_size=config.TEST_SIZE,
        random_state=config.RANDOM_STATE
    )

    print(f"Training videos: {len(train_videos)}")
    print(f"Test videos: {len(test_videos)}")

    # create datasets
    print("\nCreating datasets...")

    train_dataset = create_dataset(
        train_videos, mouth_detector, char_to_num,
        batch_size=config.BATCH_SIZE,
        max_video_length=config.MAX_VIDEO_LENGTH,
        max_label_length=config.MAX_LABEL_LENGTH,
        img_width=config.IMG_WIDTH,
        img_height=config.IMG_HEIGHT,
        shuffle=True,
        cache=False  # disable cache to avoid partial reading issues
    )

    test_dataset = create_dataset(
        test_videos, mouth_detector, char_to_num,
        batch_size=config.BATCH_SIZE,
        max_video_length=config.MAX_VIDEO_LENGTH,
        max_label_length=config.MAX_LABEL_LENGTH,
        img_width=config.IMG_WIDTH,
        img_height=config.IMG_HEIGHT,
        shuffle=False,
        cache=False  # disable cache to avoid partial reading issues
    )

    # create model
    print("\nCreating model...")
    model = create_lipnet_model(
        vocab_size=vocab_size,
        input_shape=(config.MAX_VIDEO_LENGTH, config.IMG_HEIGHT, config.IMG_WIDTH, 1),
        hidden_dim=config.HIDDEN_DIM,
        dropout_rate=config.DROPOUT_RATE
    )

    model.summary()

    # compile model
    model.compile(
        optimizer=Adam(config.LEARNING_RATE),
        loss=CTCLoss
    )

    # create callbacks
    callbacks = [
        # save best model (based on validation loss)
        ModelCheckpoint(
            os.path.join(config.CHECKPOINT_DIR, "lipnet_best.keras"),
            monitor="val_loss",
            save_best_only=True,
            verbose=1
        ),
        # save best model (based on training loss)
        ModelCheckpoint(
            os.path.join(config.CHECKPOINT_DIR, "lipnet_best_train.keras"),
            monitor="loss",
            save_best_only=True,
            verbose=1
        ),
        # learning rate scheduler
        LearningRateScheduler(scheduler),
        # early stopping
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
        # generate prediction examples
        ProduceExample(test_dataset, num_to_char, "Validation"),
        ProduceExample(train_dataset, num_to_char, "Training"),
    ]

    # train
    print("\nStarting training...")
    history = model.fit(
        train_dataset,
        epochs=config.EPOCHS,
        validation_data=test_dataset,
        callbacks=callbacks,
        verbose=1
    )

    # save final model
    model.save(os.path.join(config.CHECKPOINT_DIR, "lipnet_final.keras"))
    print(f"\nModel saved to {config.CHECKPOINT_DIR}")

    # save training history
    history_path = os.path.join(config.CHECKPOINT_DIR, "training_history.json")
    with open(history_path, 'w') as f:
        json.dump({k: [float(v) for v in vals] for k, vals in history.history.items()}, f, indent=2)

    # plot training history
    plot_training_history(history, config.CHECKPOINT_DIR)

    return model, history


def plot_training_history(history, save_dir: str):
    """plot training history"""
    plt.figure(figsize=(12, 4))

    # loss curve
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss Over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    # learning rate curve (if available)
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


# ==================== Test Function ====================

def test(config: Config, checkpoint_path: str):
    """
    Test model

    Args:
        config: configuration
        checkpoint_path: model checkpoint path
    """
    print("\n" + "="*60)
    print("LipNet Testing")
    print("="*60)

    # create vocabulary
    char_to_num, num_to_char, vocab = create_vocabulary()
    vocab_size = char_to_num.vocabulary_size()

    # create mouth detector
    mouth_detector = MouthDetector()

    # get test videos
    all_videos = get_all_videos(config.DATA_PATH)
    _, test_videos = train_test_split(
        all_videos,
        test_size=config.TEST_SIZE,
        random_state=config.RANDOM_STATE
    )

    print(f"Test videos: {len(test_videos)}")

    # create test dataset
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

    # load model
    print(f"\nLoading model from: {checkpoint_path}")
    model = tf.keras.models.load_model(
        checkpoint_path,
        custom_objects={'CTCLoss': CTCLoss}
    )

    # evaluate
    print("\nEvaluating model...")

    total_samples = 0
    total_cer = 0.0
    total_wer = 0.0
    correct_predictions = 0

    results = []

    for batch_idx, (videos, labels) in enumerate(tqdm(test_dataset)):
        # get prediction
        yhat = model.predict(videos, verbose=0)

        # CTC decoding
        decoded = tf.keras.backend.ctc_decode(
            yhat,
            [yhat.shape[1]] * videos.shape[0],
            greedy=True
        )[0][0].numpy()

        for i in range(videos.shape[0]):
            # original label
            original = tf.strings.reduce_join(
                num_to_char(labels[i])
            ).numpy().decode('utf-8').strip()

            # prediction result
            prediction = tf.strings.reduce_join(
                num_to_char(decoded[i])
            ).numpy().decode('utf-8').strip()

            # calculate CER and WER
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

            # print first 20 samples
            if total_samples <= 20:
                print(f"\nSample {total_samples}:")
                print(f"  Original:   '{original}'")
                print(f"  Prediction: '{prediction}'")
                print(f"  CER: {cer*100:.2f}%, WER: {wer*100:.2f}%")

    # calculate average metrics
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

    # save results
    results_path = os.path.join(config.CHECKPOINT_DIR, "test_results.json")
    with open(results_path, 'w') as f:
        json.dump({
            'total_samples': total_samples,
            'avg_cer': avg_cer,
            'avg_wer': avg_wer,
            'accuracy': accuracy,
            'correct_predictions': correct_predictions,
            'samples': results[:100]  # save first 100 samples
        }, f, indent=2)

    print(f"\nResults saved to {results_path}")


def calculate_cer(prediction: str, target: str) -> float:
    """Calculate Character Error Rate (CER)"""
    if len(target) == 0:
        return 1.0 if len(prediction) > 0 else 0.0

    # Edit distance
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
    """Calculate Word Error Rate (WER)"""
    pred_words = prediction.split()
    target_words = target.split()

    if len(target_words) == 0:
        return 1.0 if len(pred_words) > 0 else 0.0

    # Word-level edit distance
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


# ==================== Inference function ====================

def inference(config: Config, checkpoint_path: str, video_path: str):
    """
    Perform inference on a single video

    Args:
        config: configuration
        checkpoint_path: model checkpoint path
        video_path: video file path
    """
    print("\n" + "="*60)
    print("LipNet Inference")
    print("="*60)

    # create vocabulary
    char_to_num, num_to_char, vocab = create_vocabulary()

    # create mouth detector
    mouth_detector = MouthDetector()

    # load model
    print(f"Loading model from: {checkpoint_path}")
    model = tf.keras.models.load_model(
        checkpoint_path,
        custom_objects={'CTCLoss': CTCLoss}
    )

    # Load video
    print(f"Processing video: {video_path}")
    video_data = load_video(video_path, mouth_detector, config.IMG_WIDTH, config.IMG_HEIGHT)

    # Padding to max length
    if video_data.shape[0] < config.MAX_VIDEO_LENGTH:
        padding = tf.zeros((
            config.MAX_VIDEO_LENGTH - video_data.shape[0],
            config.IMG_HEIGHT, config.IMG_WIDTH, 1
        ))
        video_data = tf.concat([video_data, padding], axis=0)
    else:
        video_data = video_data[:config.MAX_VIDEO_LENGTH]

    # Add batch dimension
    video_data = tf.expand_dims(video_data, axis=0)

    # Predict
    yhat = model.predict(video_data, verbose=0)

    # CTC decode
    decoded = tf.keras.backend.ctc_decode(
        yhat,
        [config.MAX_VIDEO_LENGTH],
        greedy=True
    )[0][0].numpy()

    # Convert to text
    prediction = tf.strings.reduce_join(
        num_to_char(decoded[0])
    ).numpy().decode('utf-8').strip()

    print("\n" + "="*60)
    print(f"Video: {video_path}")
    print(f"Prediction: '{prediction}'")
    print("="*60)

    # If there is a corresponding label file, display the actual label
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


# ==================== Main function ====================

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description='LipNet: End-to-End Sentence-level Lipreading',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument('--mode', type=str, default='train',
                       choices=['train', 'test', 'inference'],
                       help='run mode')

    parser.add_argument('--data_path', type=str, default='./data',
                       help='dataset path')

    parser.add_argument('--checkpoint', type=str, default=None,
                       help='model checkpoint path')

    parser.add_argument('--video', type=str, default=None,
                       help='video path')

    parser.add_argument('--epochs', type=int, default=100,
                       help='number of epochs')

    parser.add_argument('--batch_size', type=int, default=8,
                       help='batch size')

    parser.add_argument('--learning_rate', type=float, default=1e-4,
                       help='learning rate')

    return parser.parse_args()


def main():
    """Main function"""
    args = parse_args()

    # create config
    config = Config()
    config.DATA_PATH = args.data_path
    config.EPOCHS = args.epochs
    config.BATCH_SIZE = args.batch_size
    config.LEARNING_RATE = args.learning_rate

    # set GPU memory growth
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
