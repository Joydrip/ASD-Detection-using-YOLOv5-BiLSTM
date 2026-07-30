# ASD-Detection-using-YOLOv5-BiLSTM
This project presents a deep learning framework for the early screening of Autism Spectrum Disorder (ASD) using facial images and video sequences.

The system combines:

- YOLOv5 for face detection
- BiLSTM for temporal feature learning
- Data preprocessing and augmentation
- Person-wise train-test splitting to eliminate data leakage

The objective is to classify facial sequences into:

- ASD
- Non-ASD

---

## Dataset

The dataset contains facial images collected from both ASD and Non-ASD individuals.

### ASD

- Video recordings
- Still facial images

### Non-ASD

- Facial images collected from publicly available datasets.

Videos are converted into approximately **15–18 uniformly spaced frames**.

---

## Preprocessing

The preprocessing pipeline consists of:

- Frame Extraction
- Face Detection
- Face Cropping
- Background Blurring
- Image Resizing (224 × 224)
- Normalization
- Data Augmentation
- Person-wise Train-Test Split

---

## Model Architecture

Input Video

↓

Frame Extraction

↓

YOLOv5 Face Detection

↓

Feature Extraction

↓

BiLSTM

↓

Dense Layers

↓

Sigmoid

↓

ASD / Non-ASD Prediction

---

## Technologies Used

- Python
- YOLOv5
- TensorFlow
- PyTorch
- OpenCV
- NumPy
- Pandas
- Google Colab

---

## Evaluation

The model was evaluated using

- Accuracy
- Precision
- Recall
- F1 Score
- Confusion Matrix
- ROC-AUC

---

## Future Work

- Larger dataset
- Eye-tracking integration
- EEG integration
- Real-time ASD screening
- Mobile deployment

---

## Authors

Joydeep Sarkar

Institute of Engineering & Management
