# Cat and Dog Image Classification Using SVM
This project implements a Support Vector Machine (SVM) to classify images of cats and dogs from the Kaggle dataset.

## Dataset
The dataset used for this project is available on Kaggle: [https://www.kaggle.com/c/dogs-vs-cats/data]

## Project Structure
train: Directory containing training images.
test1: Directory containing test images.
svm_classifier.py: Script to preprocess images, train the SVM model, and evaluate its performance.

## Requirements
--> Python 3.x
--> OpenCV
--> NumPy
--> Scikit-learn
--> Psutil

## You can install the required libraries using the following command:

bash
```
pip install opencv-python-headless numpy scikit-learn psutil
```


# Hand Gesture Recognition Model

## Overview

This repository contains a hand gesture recognition model developed as part of a task with Prodigy InfoTech. The model aims to identify and classify various hand gestures from image data, enabling intuitive human-computer interaction and gesture-based control systems.

## Dataset

The model uses the [LeapGestRecog dataset](https://www.kaggle.com/gti-upm/leapgestrecog), which includes images of different hand gestures.

### Categories
- `01_palm`
- `02_l`
- `03_fist`
- `04_fist_moved`
- `05_thumb`
- `06_index`
- `07_ok`
- `08_palm_moved`
- `09_c`
- `10_down`

## Key Features

- **Data Preparation**: Images are loaded, resized to 50x50 pixels, and normalized for training.
- **Model Architecture**: 
  - **Convolutional Layers**: Extract features from images using Conv2D layers.
  - **Pooling**: Use MaxPool2D to reduce dimensionality.
  - **Dense Layers**: Fully connected layers for classification.
  - **Dropout**: Regularization to prevent overfitting.
- **Training**: The model is trained using categorical cross-entropy loss and RMSprop optimizer. Training and validation accuracy are monitored.
- **Evaluation**: The model’s performance is evaluated on a test set, and accuracy and loss are plotted.

## Getting Started

### Prerequisites

- Python 3.x
- TensorFlow/Keras
- OpenCV
- NumPy
- Pandas
- scikit-learn
- Matplotlib

### Installation

Clone the repository and install the necessary packages:

```bash
git clone https://github.com/your-username/hand-gesture-recognition.git
cd hand-gesture-recognition
pip install -r requirements.txt
```
