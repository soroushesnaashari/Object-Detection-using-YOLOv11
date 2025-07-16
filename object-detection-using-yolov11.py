# Cell 3
pip install ultralytics

# Cell 4
import os
import cv2
import random
import pandas as pd
from pathlib import Path
from ultralytics import YOLO
import matplotlib.pyplot as plt

# Cell 6
DATASET_DIR = '/kaggle/input/vehicledetection/VehiclesDetectionDataset'
TRAIN_IMAGES = f'{DATASET_DIR}/train/images'
VALID_IMAGES = f'{DATASET_DIR}/valid/images'
TEST_IMAGES  = f'{DATASET_DIR}/test/images'
DATA_YAML    = f'{DATASET_DIR}/dataset.yaml'
VIDEO_PATH   = '/kaggle/input/vehicledetection/TestVideo/TrafficPolice.mp4'

# Cell 8
def show_samples(image_dir, title, n=6, cols=3, figsize=(12,8)):
    """
    Display random sample images from a directory
    """
    image_paths = list(Path(image_dir).glob('*.jpg'))
    samples = random.sample(image_paths, min(n, len(image_paths)))
    rows = (len(samples) + cols - 1) // cols
    plt.figure(figsize=figsize)
    for i, path in enumerate(samples):
        img = cv2.cvtColor(cv2.imread(str(path)), cv2.COLOR_BGR2RGB)
        ax = plt.subplot(rows, cols, i+1)
        ax.imshow(img)
        ax.axis('off')
    plt.suptitle(title)
    plt.tight_layout()
    plt.show()

# Cell 10
show_samples(TRAIN_IMAGES, 'Training Set Samples')
show_samples(VALID_IMAGES, 'Validation Set Samples')

# Cell 12
model = YOLO('yolo11n')  

training_results = model.train(
    data=DATA_YAML,
    imgsz=640,
    batch=16,
    epochs=100,
    project='runs/train',
    name='yolov11_vehicle',
    exist_ok=True
)

# Cell 14
from pathlib import Path

def plot_results(run_dir):
    f = Path(run_dir) / 'results.csv'
    if not f.exists():
        raise FileNotFoundError(f"No results.csv in {run_dir}")
    df = pd.read_csv(f)

    epochs = df['epoch']

    # 1) Training losses
    plt.figure()
    plt.plot(epochs, df['train/box_loss'], label='train box_loss')
    plt.plot(epochs, df['train/cls_loss'], label='train cls_loss')
    plt.plot(epochs, df['train/dfl_loss'], label='train dfl_loss')
    plt.title('Training Losses over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

    # 2) Validation losses
    plt.figure()
    plt.plot(epochs, df['val/box_loss'], label='val box_loss')
    plt.plot(epochs, df['val/cls_loss'], label='val cls_loss')
    plt.plot(epochs, df['val/dfl_loss'], label='val dfl_loss')
    plt.title('Validation Losses over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

    # 3) mAP curves
    plt.figure()
    plt.plot(epochs, df['metrics/mAP50(B)'],      label='mAP@0.5')
    plt.plot(epochs, df['metrics/mAP50-95(B)'],   label='mAP@0.5:0.95')
    plt.title('mAP over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('mAP')
    plt.legend()
    plt.show()

plot_results('runs/train/yolov11_vehicle')

# Cell 16
test_results = model.predict(
    source=TEST_IMAGES,
    imgsz=640,
    batch=16,
    save=True,
    project='runs/test',
    name='yolov11_vehicle'
)

# Cell 18
PRED_DIR = 'runs/test/yolov11_vehicle'
show_samples(PRED_DIR, 'Test Set Predictions')

# Cell 20
video_output = 'runs/video/yolov11_vehicle'
os.makedirs(video_output, exist_ok=True)
_ = model.predict(
    source=VIDEO_PATH,
    show=False,
    save=True,
    project='runs/video',
    name='yolov11_vehicle'
)
print(f"Video results saved to {video_output}")

