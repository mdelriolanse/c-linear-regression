# Linear Regression in C

This project implements a basic single-feature linear regression model from scratch using C. It demonstrates key concepts such as forward pass, backpropagation, and gradient descent.

The work is structured around the ```main.c``` entrypoint, which includes calls to the ```regression_core.c``` and ```regression_input.c``` implementation files. At compile time, you will need to link these sets of files.

```main.c``` is the program entry point. It takes two runtime arguments:
1. <num_epochs> - determines the number of iterations over the training data (epochs) the regression algorithm will perform during training.
2. <path_to_data> - path to .csv file with datapoints.

```regression_core.c``` provides all supporting training functionality, including:
1. Mean Squared Error (MSE) loss computations;
2. Forward pass training loop;
3. Backward pass training loop - computes partial derivates for weight and bias updates.

```regression_io.c``` includes functionality to validate both runtime arguments to ```main.c``` (epochs and path). This also includes csv data parser.

Macros, struct and function declarations are included in the ```regression.h``` header file.

**Multivariable linear regression implementation coming soon.**

## Installation 
Clone the repository:
```bash
git clone https://github.com/maticos-dev/c-linear-regression
```

## Passing Data
To use this program, the data must be formatted in a .csv file, where each row is structured as such:
```bash x coordinate```, ```bash y coordinate```

## Execution:
1. Compile:
   ```bash
   gcc linreg_core.c linreg_input.c main.c -o main

2. Execute
    ```bash
    ./main <number of epochs> <path to data>
