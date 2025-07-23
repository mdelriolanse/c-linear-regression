#include <assert.h>
#include <stdlib.h>
#include <stdio.h>
#include "regression_utils.h"

pRegressionParameters initialize_params() {

	// bias is one term irrespective of features.
	// no need to init using calloc.
	pRegressionParameters params = malloc(sizeof(RegressionParameters));

	if (!params) {
		fprintf(stderr, "parameters (w and b) memory allocation failed\n");
		exit(EXIT_FAILURE);
	}

	float *weights_block = (float *)calloc(NUM_FEATURES, sizeof(float));

	if (!weights_block) {
		fprintf(stderr, "weight memory allocation failed\n");
		exit(EXIT_FAILURE);
	}

	params->weights = weights_block;
	params->bias = 0;

	return params;
}

void forward_pass(pDataLoader data, pRegressionParameters params, pTrainingDiagnostics diagnostics) {
	float y_hat = 0;
	float err = 0;
	float sum_err = 0; // sum error
	float squared_err = 0;
	float weighted_err = 0; // weighted - multipled by sample.
	float mse; 
	
	float *weights = params->weights;
	float bias = params->bias;

	float *samples = data->samples;
	float *labels = data->labels;

	for (int j = 0; j < NUM_FEATURES; j++) {
		for (int i = 0; i < data->length; i++) {
			y_hat = samples[i] * weights[j] + bias; // samples[j][i] * weights[j] , where j is 
								// the number of features
			err = (y_hat - labels[i]);
			squared_err += err * err;
			sum_err += err;
			weighted_err += err * samples[i];
			// going to need to clear in multivariate case
		}
	}

	mse = squared_err / data->length;

	diagnostics->sum_err = sum_err;
	diagnostics->weighted_err = weighted_err;
	diagnostics->mse = mse;
}	

void backward_pass(pDataLoader data, pRegressionParameters params, pTrainingDiagnostics diagnostics) {

	for (int j = 0; j < NUM_FEATURES; j++) {
		float weight_grad = 2.0/data->length * (diagnostics->weighted_err);
		params->weights[j] -= LEARNING_RATE * weight_grad; 
	}

	// update bias
	float bias_grad = 2.0/data->length * (diagnostics->sum_err);
	params->bias -= LEARNING_RATE * bias_grad;
}

void clear_diagnostics(pTrainingDiagnostics diagnostics) {
	diagnostics->mse = 0;
	diagnostics->weighted_err = 0;
	diagnostics->sum_err = 0;
}

int initialize_dataloader(pDataLoader dataloader, size_t num_samples, size_t num_features) {

	assert(num_samples < MAX_SAMPLES);
	assert(num_features < MAX_FEATURES);

	dataloader->samples = malloc(num_features * num_samples * sizeof(float));
	dataloader->labels = malloc(num_features * num_samples * sizeof(float));
	dataloader->length = num_samples;

	return EXIT_SUCCESS;
}

void train(pDataLoader data, size_t epochs) {
	pRegressionParameters params = initialize_params();
	pTrainingDiagnostics diagnostics = calloc(1, sizeof(TrainingDiagnostics));

	for (int i = 0; i < epochs; i++) {
		forward_pass(data, params, diagnostics);
		backward_pass(data, params, diagnostics);
		printf("epoch: %i - mse: %e\n", i+1, diagnostics->mse);
		printf("weight: %e - bias: %e\n", params->weights[0], params->bias);
		clear_diagnostics(diagnostics);
	}

	free(params->weights);
	free(params);
	free(diagnostics);
	free(data);

}
