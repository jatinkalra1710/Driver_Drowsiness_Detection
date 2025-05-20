# Driver Drowsiness Detection System

## Overview
This project implements a real-time driver drowsiness detection system using convolutional neural networks (CNN) to classify eye states (open/closed) and mouth states (yawning/not yawning). By monitoring these facial features, the system can identify signs of fatigue and potentially prevent accidents caused by drowsy driving.

## Problem Statement
Driver fatigue is a major contributor to traffic accidents worldwide. This system addresses this critical safety challenge by providing a non-invasive method to detect early signs of drowsiness through computer vision and deep learning techniques.

## Dataset
The dataset consists of facial images categorized into four classes:
- Closed eyes
- Open eyes
- No yawn
- Yawn

Images are processed and resized to 150x150 pixels, converted to grayscale, and organized to train a multi-class classifier.

## Features
- **Multi-class classification:** Detects four distinct states (closed eyes, open eyes, no yawn, yawn)
- **Data augmentation:** Implements horizontal/vertical flipping and brightness/contrast adjustments for improved robustness
- **CNN architecture:** Uses a three-layer convolutional neural network with dropout regularization
- **Comprehensive evaluation:** Includes confusion matrix analysis and visualization of model performance

## Model Architecture
```
Model: Sequential
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
Conv2D (32, (3, 3))          (None, 148, 148, 32)      320       
MaxPooling2D (2, 2)          (None, 74, 74, 32)        0         
Conv2D (64, (3, 3))          (None, 72, 72, 64)        18,496    
MaxPooling2D (2, 2)          (None, 36, 36, 64)        0         
Conv2D (128, (3, 3))         (None, 34, 34, 128)       73,856    
MaxPooling2D (2, 2)          (None, 17, 17, 128)       0         
Flatten                      (None, 36992)             0         
Dense (256)                  (None, 256)               9,470,208 
Dropout (0.5)                (None, 256)               0         
Dense (4)                    (None, 4)                 1,028     
=================================================================
Total params: 9,563,908
Trainable params: 9,563,908
Non-trainable params: 0
```

## Model Performance
- Overall accuracy: ~94% across all four classes
- Class-specific accuracy:
  - Closed eyes: ~96%
  - Open eyes: ~92% 
  - No yawn: ~93%
  - Yawn: ~95%

## Visualization
The project includes visualization tools for:
- Training and validation accuracy/loss curves
- Sample predictions with correct/incorrect classifications
- Confusion matrix for detailed performance analysis

## Future Improvements
- Integration with real-time video streams
- Deployment on edge devices for in-vehicle monitoring
- Additional fatigue indicators (head position, blinking frequency)
- Alert system with customizable thresholds
