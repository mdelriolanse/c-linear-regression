#ifndef UTILS_H
#define UTILS_H

#include <stddef.h>

#define EXIT_SUCCESS 0
#define EXIT_FAILURE 1
#define LEARNING_RATE 0.0001
#define MAX_EPOCHS 2500
#define MIN_EPOCHS 1
#define MAX_SAMPLES 1000
#define MAX_FEATURES 10
#define PATH_BUF_LEN 256

//STRUCT DECLARATIONS
typedef struct {
	float *weights;
	float bias;
}RegressionParameters, *pRegressionParameters;

typedef struct {
	float **samples;  // 2D array: samples[feature][sample_index]
	float *labels;
	size_t length;
	size_t num_features;
}DataLoader, *pDataLoader;

typedef struct {
	float mse;
	float sum_err;
	float *weighted_err;  // Array for each feature's weighted error
}TrainingDiagnostics, *pTrainingDiagnostics;

//FUNCTION DECLARATIONS

//initializations
pRegressionParameters initialize_params(size_t num_features);
int initialize_dataloader(pDataLoader dataloader, size_t num_samples, size_t num_features);

//validations
void validate_initialization(); // check num_features = num_labels
size_t validate_epochs(char *nepochs);
void validate_path(char *buffer, char *path);

//training
void train(pDataLoader data, size_t epochs);
void forward_pass(pDataLoader data, pRegressionParameters params, pTrainingDiagnostics diagnostics);
void backward_pass(pDataLoader data, pRegressionParameters params, pTrainingDiagnostics diagnostics);
void clear_diagnostics(pTrainingDiagnostics diagnostics, size_t num_features);

//csv parser
int csv_parser(char *path, pDataLoader dataloader, size_t nsamples, size_t nfeatures);

#endif
