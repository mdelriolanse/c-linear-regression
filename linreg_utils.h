#ifndef UTILS_H
#define UTILS_H

#include <stddef.h>

#define EXIT_SUCCESS 0
#define EXIT_FAILURE 1
#define NUM_FEATURES 1
#define LEARNING_RATE 0.01
#define MAX_EPOCHS 2500
#define MIN_EPOCHS 1

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

//function declarations
pRegressionParameters initialize_params();
void train(pDataLoader data, size_t epochs);
void validate_initialization(); // check that weights are same size as features -- not necessary anymore.
			       // num samples == num labels
void forward_pass(pDataLoader data, pRegressionParameters params, pTrainingDiagnostics diagnostics);
void backward_pass(pDataLoader data, pRegressionParameters params, pTrainingDiagnostics diagnostics);

#endif
