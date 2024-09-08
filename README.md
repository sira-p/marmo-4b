# Fine-Tuning Pose Estimation and Instance Segmentation on Marmoset Data

This Google Colab notebook provides a comprehensive workflow for fine-tuning a pose estimation model and performing instance segmentation using marmoset data. The process utilizes Ultralytics' YOLOv8 model for pose estimation and integrates the Segment Anything Model (SAM) to generate masks for each frame, enhancing the segmentation by combining masks and bounding boxes.

## Overview

The notebook is designed to perform the following tasks:

1. Downloading and Preparing Marmoset Data: Automatically downloads the dataset from the [DeepLabCut Benchmark](https://benchmark.deeplabcut.org/datasets.html).
2. Fine-Tuning YOLOv8 for Pose Estimation: Leverages the Ultralytics YOLOv8 model for pose estimation on the marmoset dataset.
3. Generating Masks with SAM: Applies SAM (Segment Anything Model) to generate masks for each frame.
4. YOLO Instance Segmentation: Combines the generated masks with bounding boxes from YOLO for improved instance segmentation.
5. Advanced: Joint Pose Estimation and Instance Segmentation: Optionally combines both tasks into a unified model for better performance.
6. Visualization and Evaluation: Provides tools for visualizing and evaluating the results from pose estimation and segmentation.

## Getting Started

### Prerequisites

- A Google account to access Google Colab.
- Basic knowledge of Python, deep learning, and computer vision.
- Familiarity with Ultralytics YOLO models and the Segment Anything Model (SAM).

### Dependencies
The depedencies and prerequisites are being install via console command in the notebook.

## Notebook Structure
1. Transfer Learning from Pre-trained Pose Detection Model
Using marmosets dataset from deeplabcut benchmark to fine tune YOLOv8 pose estimation pretrained model. The marmoset dataset labels are      format to YOLOv8 format using deeplabcut2yolo packages (credits to Sira).
2. SAM is used to create a segmentation mask with keypoints and bounding box from part 1, the boudning box is estimated using the xmax, xmin, ymax, and ymin of the keypoints. Later, a function is defined to combine both models predictions into one function for ease of use. This part relies on the pretrained models from part 1 labelled 'marmo-pose-one-class.pt' and 'marmo-pose-two-class.pt', these models are trained for 120 epochs with marmoset dataset from deeplabcut, the one class model is recommeded as it offers better prediction. This part is self contained and does not require the user to initiate the cells in part 1.
3. An webapp interface is created using Gradio for ease of use. (Dependent on part 2 to load models and dependencies)
