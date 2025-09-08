# Multidimensional Linear Regression in C

A high-performance implementation of linear regression with support for multiple input features, written in C. This project demonstrates gradient descent optimization for both one-dimensional and multidimensional regression problems.

## Features

- ✅ **Multidimensional Support**: Handle 1 to 10 input features
- ✅ **Gradient Descent**: Efficient optimization with configurable learning rate
- ✅ **Memory Efficient**: Optimized 2D array storage for large datasets
- ✅ **CSV Data Loading**: Easy data import from CSV files
- ✅ **Real-time Training**: Monitor convergence with epoch-by-epoch output
- ✅ **Robust Error Handling**: Comprehensive validation and error checking

## Project Structure

```
c-linear-regression/
├── main.c                 # Main program entry point
├── regression_core.c      # Core regression algorithms
├── regression_io.c        # CSV parsing and I/O operations
├── regression_utils.h     # Header file with data structures and function declarations
├── singledim_data.csv     # 1D sample dataset (100 samples)
├── multidim_data.csv 	   # 2D sample dataset (100 samples, 2 features)
└── README.md             
```

## Quick Start

### Compilation

```bash
gcc -o linear_regression main.c regression_core.c regression_io.c -Wall -Wextra
```

### Usage

```bash
./linear_regression <num_epochs> <num_features> <num_samples> <data_file_path>
```

### Examples

**1D Linear Regression:**
```bash
./linear_regression 100 1 100 singledim_data.csv
```

**2D Linear Regression:**
```bash
./linear_regression 100 2 100 multidim_data.csv
```

## Data Format

The program expects CSV files with the following format:
```
feature1,feature2,...,featureN,output
```

### Example Data Files

**1D Data (sample_data.csv):**
```
1,2.14
2,3.88
3,6.11
4,7.95
5,10.02
...
```

**2D Data (multidim_sample_data.csv):**
```
1,2,3.14
2,3,5.88
3,4,8.11
4,5,10.95
5,6,13.02
...
```

## Algorithm Details

### Mathematical Model

For multidimensional linear regression, the model predicts:
```
y_hat = w₁x₁ + w₂x₂ + ... + wₙxₙ + b
```

Where:
- `y_hat` is the predicted output
- `w₁, w₂, ..., wₙ` are the feature weights
- `x₁, x₂, ..., xₙ` are the input features
- `b` is the bias term

### Gradient Descent

The algorithm uses gradient descent to minimize the mean squared error (MSE):

**Forward Pass:**
- Compute predictions for all samples
- Calculate MSE: `MSE = (1/n) * Σ(y_hat - y)²`

**Backward Pass:**
- Update weights: `wᵢ = wᵢ - α * (2/n) * Σ(y_hat - y) * xᵢ`
- Update bias: `b = b - α * (2/n) * Σ(y_hat - y)`

Where `α` is the learning rate (default: 0.0001).

## Configuration

### Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `LEARNING_RATE` | 0.0001 | Gradient descent step size |
| `MAX_EPOCHS` | 2500 | Maximum training iterations |
| `MIN_EPOCHS` | 1 | Minimum training iterations |
| `MAX_SAMPLES` | 1000 | Maximum number of data samples |
| `MAX_FEATURES` | 10 | Maximum number of input features |

### Data Structures

```c
typedef struct {
    float *weights;     // Array of feature weights
    float bias;         // Bias term
} RegressionParameters;

typedef struct {
    float **samples;    // 2D array: samples[feature][sample_index]
    float *labels;      // Output values
    size_t length;      // Number of samples
    size_t num_features; // Number of input features
} DataLoader;

typedef struct {
    float mse;          // Mean squared error
    float sum_err;      // Sum of errors
    float *weighted_err; // Weighted errors for each feature
} TrainingDiagnostics;
```

## Sample Output

```
Initializing 100 epochs
Using 2 features
Using 100 samples
multidim_sample_data.csv
epoch: 1 - mse: 1.604226e+04
weights: w0=1.472700e+00 w1=1.495039e+00 - bias: 2.233916e-02
epoch: 2 - mse: 2.254612e+03
weights: w0=9.218054e-01 w1=9.362061e-01 - bias: 1.440068e-02
epoch: 3 - mse: 3.268029e+02
weights: w0=1.127588e+00 w1=1.145372e+00 - bias: 1.778380e-02
...
epoch: 20 - mse: 1.341520e+01
weights: w0=1.069073e+00 w1=1.090845e+00 - bias: 2.177159e-02
```

## Performance

- **Memory Usage**: O(features × samples) for data storage
- **Time Complexity**: O(epochs × features × samples) per training run
- **Convergence**: Typically converges within 50-200 epochs for well-conditioned data

## Error Handling

The program includes comprehensive error checking for:
- Invalid command line arguments
- File I/O errors
- Memory allocation failures
- CSV format validation
- Parameter bounds checking

## Development

### Building with Debug Information

```bash
gcc -g -o linear_regression main.c regression_core.c regression_io.c -Wall -Wextra
```

### Running with Valgrind (Linux/macOS)

```bash
valgrind --leak-check=full ./linear_regression 100 2 100 multidim_data.csv
```

## License

This project is open source. See the LICENSE file for details.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## Future Enhancements

- [ ] Support for different loss functions (MAE, Huber loss)
- [ ] Regularization (L1, L2)
- [ ] Feature scaling/normalization
- [ ] Cross-validation support
- [ ] Model persistence (save/load trained models)
- [ ] GPU acceleration with CUDA
- [ ] Python bindings

## Troubleshooting

### Common Issues

**Program exits without output:**
- Check that the number of samples matches your CSV file
- Verify CSV format is correct
- Ensure file path is accessible

**Weights explode to infinity:**
- Reduce learning rate in `regression_utils.h`
- Check for data normalization issues
- Verify input data quality

**Poor convergence:**
- Increase number of epochs
- Adjust learning rate
- Check data preprocessing

### Getting Help

If you encounter issues:
1. Check the error messages
2. Verify your data format
3. Test with the provided sample datasets
4. Review the configuration parameters
