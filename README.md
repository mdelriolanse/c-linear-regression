# 🧮 Linear Regression in C

This project implements a basic single-feature linear regression model from scratch using C. It demonstrates key concepts such as forward pass, backpropagation, and gradient descent.

## 📦 Overview

- **Language:** C
- **Model:** Simple Linear Regression
- **Features:** Single feature input (`NUM_FEATURES = 1`)
- **Learning Strategy:** Mean Squared Error (MSE) loss, batch gradient descent
- **Use Case:** Fits a line to data following the pattern `y = 2x + 1`

## 📁 Files

### `main.c`
Contains:
- Struct definitions for model parameters, training data, and diagnostics
- Functions for initializing parameters, forward and backward pass computations
- Training loop with weight updates via gradient descent

## 🚀 How It Works

1. **Initialize Parameters**
   - Zero weights and bias
2. **Load Data**
   - Hardcoded training dataset (`x = [1, 2, 3, 4, 5]`, `y = [3, 5, 7, 9, 11]`)
3. **Training Loop**
   - Perform forward pass to compute predictions and errors
   - Use backward pass to update weights and bias
   - Print MSE for each epoch

## 🧠 Core Components

### Structures

| Struct | Description |
|--------|-------------|
| `RegressionParameters` | Holds the weights and bias |
| `DataLoader` | Holds input samples and labels |
| `TrainingDiagnostics` | Stores metrics: MSE, total error, weighted error |

### Functions

| Function | Purpose |
|---------|---------|
| `initialize_params()` | Allocates and initializes model weights and bias |
| `forward_pass()` | Calculates predictions, errors, and MSE |
| `backward_pass()` | Applies gradient descent to update parameters |
| `clear_diagnostics()` | Resets metrics for next epoch |
| `train_linreg()` | Coordinates training loop |

## 🔧 How to Run

1. Compile:
   ```bash
   gcc main.c -o linreg

2. Execute
    ```bash
    ./linreg
