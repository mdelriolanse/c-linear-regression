#ifndef UTILS_H
#define UTILS_H

#include <stddef.h>

#define EXIT_SUCCESS 0
#define EXIT_FAILURE 1
#define NUM_FEATURES 1
#define LEARNING_RATE 0.01
#define MAX_EPOCHS 2500
#define MIN_EPOCHS 1
#define MAX_SAMPLES 1000
#define MAX_FEATURES 10 // for now only one really.

//STRUCT DECLARATIONS
typedef struct {
	float *weights;
	float bias;
}RegressionParameters, *pRegressionParameters;

typedef struct {
	float *samples;
	float *labels;
	size_t length;
}DataLoader, *pDataLoader;

typedef struct {
	float mse;
	float sum_err;
	float weighted_err;
}TrainingDiagnostics, *pTrainingDiagnostics;

//FUNCTION DECLARATIONS

//initializations
pRegressionParameters initialize_params();
void initialize_dataloader(pDataLoader dataloader, size_t num_samples, size_t num_features);

//validations
void validate_initialization(); // check num_features = num_labels
void validate_epochs(char *nepochs);

//training
void train(pDataLoader data, size_t epochs);
void forward_pass(pDataLoader data, pRegressionParameters params, pTrainingDiagnostics diagnostics);
void backward_pass(pDataLoader data, pRegressionParameters params, pTrainingDiagnostics diagnostics);

//csv parser
void csv_parser(char *path, pDataLoader dataloader, size_t nsamples, size_t nfeatures);

#endif
