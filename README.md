<h1 align="center">
Convolutional Neural Network (CNN) for MNIST Image Classification
</h1>

This Python code implements a Convolutional Neural Network (CNN) for classifying hand-written digit images from the MNIST database. The goal is to accurately identify the digits from 0 to 9 present in the images.

## Overview

The neural network implementation comprises the following components:

***Data Loading and Preprocessing***
- Loads the MNIST dataset, preprocesses images, and prepares them for training.

***Convolutional Neural Network***
- Defines the architecture of the CNN, including convolutional layers, pooling layers, fully connected layers, and output layers.
- Trains the network to recognize and classify digit images.

***Training and Evaluation***
- Manages the training process by optimizing the network's weights using backpropagation and gradient descent.
- Evaluates the model's performance on a test set, calculating accuracy and loss metrics.

## Usage

The code can be utilized by setting parameters such as the network architecture, hyperparameters like **learning rate**, **batch size**, **number of epochs**, etc.

The CNN iteratively learns from the MNIST dataset, adjusting its internal parameters through forward and backward propagation to accurately classify hand-written digits.

You can try it out in COLAB:

[COLAB LINK](https://colab.research.google.com/drive/15nO-fPEy0w4TqM-P68pySgSVTexn-0zF?usp=sharing)

## CODE
Libraries Import:
```python
import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
