# 🧮 Linear Regression in C

This project implements a basic single-feature linear regression model from scratch using C. It demonstrates key concepts such as forward pass, backpropagation, and gradient descent.

**Multivariable linear regression implementation coming soon.**
## 📦 Overview

- **Language:** C
- **Model:** Simple Linear Regression
- **Features:** Single feature input (`NUM_FEATURES = 1`) - Multivariable coming soon.
- **Learning Strategy:** Mean Squared Error (MSE) loss, batch gradient descent
- **Use Case:** Fits a line to data any inputted .csv data.

## 📁 Files

### `main.c`
Contains:
- Entry point for linear regression algorithm.
- Calls to supplementary modules (more below).

### `linreg_core.c`
- Struct definitions for model parameters, data loader, and diagnostics
- Functions for initializing parameters, forward and backward pass computations
- Training loop with weight updates via gradient descent

### `linreg_input.c`
- CSV file parser; functionality to parse user-inputted csv file.
- Validation function for user inputted .csv file path containing data.
- Epoch validation function to ensure that a valid unsigned integer type is inputted.

## 🚀 How It Works

1. **Initialize Parameters**
   - Zero weights and bias
2. **Load Data**
   - User inputted training dataset. Pass path to .csv file containing datapoints.
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

1. Compile (all included in make file):
   ```bash
   gcc linreg_core.c linreg_input.c main.c -o main

2. Execute
    ```bash
    ./main
