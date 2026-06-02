#ifndef UTILS_H
#define UTILS_H

#include <stddef.h>
#include <errno.h>

#define EXIT_SUCCESS 0
#define EXIT_FAILURE 1

#define LEARNING_RATE 0.0001
#define MAX_LR 1
#define MIN_LR 1e-12

#define MAX_EPOCHS 2500
#define MIN_EPOCHS 1

#define MAX_SAMPLES 1000
#define MAX_FEATURES 10
#define PATH_BUF_LEN 256

// Error Handling Macro
#define DIE(msg) do { \
	fprintf(stderr, "%s:%d: %s: ", __FILE__, __LINE__, __func__); \
	perror(msg); \
	exit(EXIT_FAILURE); \
} while (0)

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

typedef struct {
	size_t epochs;
	float learning_rate;
}TrainingConfig, *pTrainingConfig;

//FUNCTION DECLARATIONS

//initializations
pRegressionParameters initialize_params(size_t num_features);
int initialize_dataloader(pDataLoader dataloader, size_t num_samples, size_t num_features);
int initialize_training_config(pTrainingConfig cfg, size_t epochs, float learning_rate);

//validations
void validate_initialization(); // check num_features = num_labels
size_t validate_epochs(char *nepochs);
void validate_path(char *buffer, char *path);
float validate_learning_rate(char *learning_rate);

//training
void train(pDataLoader data, pTrainingConfig cfg);
void forward_pass(pDataLoader data, pRegressionParameters params, pTrainingDiagnostics diagnostics);
void backward_pass(pDataLoader data, pRegressionParameters params, pTrainingDiagnostics diagnostics, float learning_rate);
void clear_diagnostics(pTrainingDiagnostics diagnostics, size_t num_features);

//csv parser
int csv_parser(char *path, pDataLoader dataloader, size_t nsamples, size_t nfeatures);

#endif
