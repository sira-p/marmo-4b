##README

# **Fine-Tuning Pose Estimation and Instance Segmentation on Marmoset Data**

This Google Colab notebook provides a complete workflow to fine-tune a pose estimation model and perform instance segmentation using marmoset data. The process involves using Ultralytics' YOLOv8 for pose estimation, and the Segment Anything Model (SAM) for mask generation, to enhance segmentation accuracy by combining masks and bounding boxes.

## **Overview**

The notebook is structured as follows:

1. **Downloading and Preparing Marmoset Data**: Automatically downloads the dataset from the [DeepLabCut Benchmark](https://benchmark.deeplabcut.org/datasets.html).
2. **Fine-Tuning YOLOv8 for Pose Estimation**: Utilizes Ultralytics YOLOv8 for pose estimation on the marmoset dataset.
3. **Generating Masks with SAM**: Uses SAM (Segment Anything Model) to create masks for each frame.
4. **YOLO Instance Segmentation**: Integrates the generated masks with bounding boxes from YOLO for instance segmentation.
5. **Advanced: Joint Pose Estimation and Instance Segmentation**: Optionally combines both tasks into a unified model for improved performance.
6. **Visualization and Evaluation**: Visualizes and evaluates the results from pose estimation and segmentation.

## **Getting Started**

### **Prerequisites**

- Google account for accessing Google Colab.
- Basic knowledge of Python, deep learning, and computer vision.
- Familiarity with Ultralytics YOLO models and the Segment Anything Model (SAM).

### **Dependencies**

Ensure the following Python libraries are installed:

- ultralytics for YOLO models
- segment-anything for SAM model usage
- opencv-python-headless for image processing
- torch for deep learning support
- matplotlib for visualizations
- pandas for data handling

Install dependencies by running:

python
!pip install ultralytics segment-anything opencv-python-headless torch matplotlib pandas

## **Data Preparation**

1. **Download Marmoset Data**: Automatically download the dataset using:

python
!wget -O marmoset_data.zip https://benchmark.deeplabcut.org/datasets/marmoset.zip
!unzip marmoset_data.zip -d ./data

2. **Load and Preprocess Data**: Load and prepare images for model training.

python
import os
import cv2
import numpy as np

data_path = './data/marmoset/'
images = []
for filename in os.listdir(data_path):
    img = cv2.imread(os.path.join(data_path, filename))
    if img is not None:
        images.append(img)

## **Notebook Structure**

### **Section 1: Data Preparation**

- **Mount Google Drive**: Access the dataset from Google Drive.

python
from google.colab import drive
drive.mount('/content/drive')

- **Download and Preprocess Data**: Download marmoset data and prepare it for model training.

### **Section 2: Fine-Tuning YOLOv8 for Pose Estimation**

- **Initialize YOLOv8 Model**: Load the YOLOv8 model for pose estimation.

python
from ultralytics import YOLO

# Load YOLOv8 model for pose estimation
model = YOLO('yolov8n-pose.pt')

- **Train the Model**: Fine-tune the YOLOv8 model on the marmoset dataset.

python
model.train(data='marmoset_data.yaml', epochs=50, imgsz=640)

### **Section 3: Generating Masks Using SAM**

- **Apply SAM for Mask Generation**: Use the Segment Anything Model (SAM) to generate masks for each frame.

python
from segment_anything import SAM, SamPredictor

# Load SAM model
sam_model = SAM('sam_vit_b')
sam_predictor = SamPredictor(sam_model)

# Generate masks
masks = []
for img in images:
    mask = sam_predictor.predict(img)
    masks.append(mask)

### **Section 4: Instance Segmentation with YOLO**

- **Combine Masks and Bounding Boxes**: Integrate SAM-generated masks with YOLO bounding boxes for instance segmentation.

python
# Perform instance segmentation using YOLOv8
results = model.predict(source=masks, conf=0.25)

# Visualize results
model.show(results)

### **Section 5: Joint Pose Estimation and Instance Segmentation [Advanced]**

- **Joint Training**: Train a model that performs both pose estimation and instance segmentation.

python
# Joint training for pose estimation and instance segmentation
joint_model = YOLO('yolov8n-pose-instance-segmentation.pt')
joint_model.train(data='marmoset_data.yaml', epochs=100, imgsz=640)

### **Section 6: Visualization and Evaluation**

- **Evaluate Results**: Visualize and evaluate the performance of the models.

python
import matplotlib.pyplot as plt

# Visualization of results
for i, result in enumerate(results):
    plt.imshow(result.img)
    plt.title(f"Result {i}")
    plt.show()
