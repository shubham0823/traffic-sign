# Traffic Sign Recognition System

## Overview

This project implements a Traffic Sign Recognition System using a Convolutional Neural Network (CNN). The model is trained on a dataset containing 40,000 images of 40 different traffic signs. It uses data augmentation and advanced image processing techniques to achieve high accuracy in traffic sign detection and classification.

## Features

Detects and classifies 40 types of traffic signs.

Trained using a CNN model for high accuracy.

Uses 40,000 labeled traffic sign images for training.

Implements data augmentation to improve model generalization.

Easy integration with camera feeds for real-time recognition.



---

## Installation and Setup

### Prerequisites

Ensure the following libraries are installed in your environment:

Python 3.x

NumPy

OpenCV

Matplotlib

TensorFlow / Keras

Scikit-learn

Pandas

Joblib


#### Install the required packages using pip:

pip install numpy opencv-python matplotlib tensorflow keras scikit-learn pandas joblib

Dataset

1. Download a traffic sign dataset (e.g., GTSRB or other labeled datasets).


2. Extract the dataset and place it in the project directory.


3. Update the dataset path in your code.




---




---

## Code Explanation

1. Import Libraries

The required libraries for preprocessing, training, and evaluation are imported:

import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.optimizers import Adam
from keras.utils.np_utils import to_categorical
from keras.preprocessing.image import ImageDataGenerator
import cv2
from sklearn.model_selection import train_test_split
import os
import random

2. Preprocessing

Resize Images: All images are resized to a consistent shape for CNN input.

Normalize Data: Pixel values are normalized to improve model performance.

One-hot Encoding: Labels are one-hot encoded for classification.


3. Data Augmentation

Use ImageDataGenerator for data augmentation to enhance model generalization:

datagen = ImageDataGenerator(
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    zoom_range=0.1
)
datagen.fit(X_train)

4. CNN Model

A sequential CNN model is designed for image classification:

model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
    MaxPooling2D(pool_size=(2, 2)),
    Dropout(0.2),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(40, activation='softmax')
])

model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])

5. Training

Split the data into training and validation sets:

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
history = model.fit(datagen.flow(X_train, y_train), validation_data=(X_test, y_test), epochs=20, batch_size=32)

6. Evaluation

Evaluate the model's performance on the test set:

score = model.evaluate(X_test, y_test, verbose=0)
print(f"Test Accuracy: {score[1] * 100:.2f}%")

7. Real-Time Detection

Integrate the model with a camera feed to detect traffic signs in real time:

cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()
    processed_frame = preprocess_image(frame)
    prediction = model.predict(processed_frame)
    cv2.imshow("Traffic Sign Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()


---

## Results

Achieved high accuracy on the test set.

Successfully detects and classifies traffic signs in real time.



---

## Future Improvements

1. Use a larger and more diverse dataset for better generalization.


2. Optimize the CNN architecture for faster inference.


3. Deploy the model on edge devices like Raspberry Pi for real-time applications.




---

## Acknowledgments

Frameworks: Keras, TensorFlow, OpenCV


