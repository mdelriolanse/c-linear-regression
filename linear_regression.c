#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>

#define EXIT_SUCCESS 0
#define EXIT_FAILURE 1
#define NUM_FEATURES 1
#define LEARNING_RATE 0.01

//struct declarations
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
	float sse;
}TrainingDiagnostics, *pTrainingDiagnostics;

//function declarations
pRegressionParameters initialize_params();
void train_linreg(pDataLoader data, size_t epochs);
void validate_initialization(); // check that weights are same size as features -- not necessary anymore.
			       // num samples == num labels
void forward_pass(pDataLoader data, pRegressionParameters params, pTrainingDiagnostics *diagnostics);
void backward_pass(pDataLoader data, pRegressionParameters params, pTrainingDiagnostics *diagnostics);

int main() {
	float x[] = {1, 2, 3, 4, 5};
	float y[] = {3, 5, 7, 9, 11};

	pDataLoader data_loader = malloc(sizeof(DataLoader));
	data_loader->samples = x;
	data_loader->labels = y;
	data_loader->length = 5;

	train_linreg(data_loader, 5);

	return EXIT_SUCCESS;
}

pRegressionParameters initialize_params() {

	// bias is one term irrespective of features.
	// no need to init using calloc.
	pRegressionParameters params = malloc(sizeof(RegressionParameters));
	float *weights_block = (float *)calloc(NUM_FEATURES, sizeof(float));

	if (!weights_block) {
		fprintf(stderr, "weight memory allocation failed\n");
		exit(EXIT_FAILURE);
	}

	params->weights = weights_block;
	params->bias = 0;

	return params;
}

void forward_pass(pDataLoader data, pRegressionParameters params, pTrainingDiagnostics *diagnostics) {
	float y_hat;
	float sse; // sum squared error
	float mse; 
	
	float *weights = params->weights;
	float bias = params->bias;

	float *samples = data->samples;
	float *labels = data->labels;

	for (int i = 0; i < data->length; i++) {
		y_hat = samples[i] * weights[i];
		sse += (labels[i] - y_hat) * (labels[i] - y_hat);
	}

	mse = sse / data->length;

	(*diagnostics)->sse = sse;
	(*diagnostics)->mse = mse;
}	

void backward_pass(pDataLoader data, pRegressionParameters params, pTrainingDiagnostics *diagnostics) {
	// update bias
	float bias_grad = 2.0/data->length * ((*diagnostics)->sse);
	params->bias -= LEARNING_RATE * bias_grad;

	for (int i = 0; i < data->length; i++) {
		float weight_grad = data->samples[i] * bias_grad;
		params->weights[i] -= LEARNING_RATE * weight_grad; 
	}
}

void clear_diagnostics(pTrainingDiagnostics *diagnostics) {
	(*diagnostics)->mse = 0;
	(*diagnostics)->sse = 0;
}

void train_linreg(pDataLoader data, size_t epochs) {
	pRegressionParameters params = initialize_params();
	pTrainingDiagnostics diagnostics = malloc(sizeof(TrainingDiagnostics));

	for (int i = 0; i < epochs; i++) {
		forward_pass(data, params, &diagnostics);
		backward_pass(data, params, &diagnostics);
		printf("epoch: %i - mse: %f\n", i, diagnostics->mse);
		clear_diagnostics(&diagnostics);
	}

}

// g
