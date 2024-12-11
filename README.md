# MNIST Handwritten Digit Classification with Custom Neural Networks

## Overview
This project implements a neural network from scratch in Python to classify handwritten digits from the **MNIST dataset**. It demonstrates:
- Training with different activation functions (`ReLU` and `tanh`).
- Mini-batch training for improved performance.
- L2 regularization to reduce overfitting.
- Manual computation of gradients and backpropagation.

---

## Dataset
The [MNIST dataset](http://yann.lecun.com/exdb/mnist/) is used, which includes:
- **Training Data**: 60,000 images of handwritten digits (28x28 pixels).
- **Test Data**: 10,000 images of handwritten digits (28x28 pixels).

### Dataset Files:
- `train-images-idx3-ubyte.gz`: Training images.
- `train-labels-idx1-ubyte.gz`: Labels for training images.
- `t10k-images-idx3-ubyte.gz`: Test images.
- `t10k-labels-idx1-ubyte.gz`: Labels for test images.

---

## Features
1. **Custom Neural Networks**:
   - `tanh` and `ReLU` activation functions.
   - Mini-batch training.
   - L2 regularization.

2. **Hyperparameter Tuning**:
   - Hidden layer sizes: 300, 500, 1000.
   - Learning rates: 0.01, 0.05, 0.09.
   - Mini-batch sizes: 10, 50, 100.
   - Regularization (lambda values): 0.001, 0.01.

3. **Outputs**:
   - Test accuracy and incorrectly classified patterns are logged to result files.

---

## File Structure
- **main.py**: The main script that trains, evaluates, and logs results.
- **Dataset Files**: MNIST data files (`*.gz`).
- **Result Files**:
  - `relu_results.txt`: Results for `ReLU` activation function.
  - `tanh_results.txt`: Results for `tanh` activation function.
  - `tanh_with_batch_results.txt`: Results for mini-batch training with `tanh`.
  - `tanh_with_batch_reg_results_lambda_*.txt`: Results for regularization.

---

## Requirements
- Python 3.8+
- Required libraries:
  - `numpy`
