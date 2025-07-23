## Object Detection using YOLOv11
[![](Videos/TrafficPolice.mp4)](https://www.kaggle.com/code/soroushesnaashari/object-detection-using-yolov11/output)

### Overview
This project demonstrates how to train and deploy the Ultralytics YOLOv11‑nano model on a custom vehicle detection dataset. Using this lightweight, state‑of‑the‑art architecture, we achieve fast inference while maintaining high detection accuracy. After training on still images, the notebook walks through inference on a held‑out test set and on a traffic video.

<br>

### Project Flow

1. **Environment & Dependencies**  
   - Install the Ultralytics YOLO package.  
   - Import essential libraries: Matplotlib, Pandas, Pathlib, etc.

2. **Data Preparation**  
   - Define paths to training, validation and test image folders, plus the dataset YAML file (`dataset.yaml`).  
   - Specify a sample video for real‑time inference (`TrafficPolice.mp4`).

3. **Visualization Utility**  
   - Define `show_samples()` to display random images from any directory.

4. **Exploratory Sampling**  
   - Render random samples from the **training** and **validation** splits to verify data integrity.

5. **Model Initialization & Training**  
   ```
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
   ```
   - Uses YOLOv11-nano (`yolo11n`) checkpoint.
   - Trains for 100 epochs on 640×640 images with batch size 16.
   - Output logs, model weights and training curves are saved under `runs/train/yolov11_vehicle/`.

6. **Test‑Set Inference**
   ```
   test_results = model.predict(
       source=TEST_IMAGES,
       imgsz=640,
       batch=16,
       save=True,
       project='runs/test',
       name='yolov11_vehicle'
   )
   ```
   - Runs detector on held‑out test images.
   - Saves annotated images in `runs/test/yolov11_vehicle/`.

7. **Prediction Visualization**
   - Display a grid of sample detection results (`show_samples('runs/test/yolov11_vehicle', 'Test Set Predictions')`) for qualitative assessment.

8. **Video Inference**
   ```
   os.makedirs('runs/video/yolov11_vehicle', exist_ok=True)
   model.predict(
       source=VIDEO_PATH,
       show=False,
       save=True,
       project='runs/video',
       name='yolov11_vehicle'
   )
   ```
   - Processes `TrafficPolice.mp4`, saving the output video with overlaid bounding boxes to `runs/video/yolov11_vehicle/`.

<br>

### Key Features

- Lightweight YOLOv11‑nano
  - Fast training and inference on limited hardware while maintaining strong object detection performance.

- End‑to‑End Pipeline
  - From raw image visualization and model training to batch‑mode inference on images and videos.

- Flexible Inference API
  - Single `model.predict()` call handles images folder, individual files or video streams.

- Visualization Helpers
  - Built‑in Matplotlib utility for quick sampling and qualitative checks at every stage.

- Reproducible Runs
  - All outputs (weights, logs, predictions) are organized under `runs/{train|test|video}/yolov11_vehicle/` with `exist_ok=True`.

<br>

### Results

- Training Logs
  - Loss curves, mAP snapshots and model checkpoints are available in `runs/train/yolov11_vehicle/`.

- Sample Detections


- Inference Statistics
  - Image Batch Inference: ~ 16 images per batch at 640×640.
  - Video Processing: Saved to `runs/video/yolov11_vehicle/TrafficPolice.mp4`.
 
<br>

### Repository Contents
- **`Object-Detection-using-YOLOv11.ipynb`**: Jupyter Notebook with full code, visualizations and explanations.
- **`Data`:** Contains the [Original Dataset](https://www.kaggle.com/datasets/alkanerturan/vehicledetection) and you can find it in the repository as the Dataset folder.
- **`README.md`:** Project documentation.

<br>

### How to Contribute
Contributions are welcome! If you'd like to improve the project or add new features:

1. **Fork the repository.**
2. **Create a new branch.**
3. **Make your changes and submit a pull request.**
