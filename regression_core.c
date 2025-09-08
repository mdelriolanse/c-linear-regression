#include <assert.h>
#include <stdlib.h>
#include <stdio.h>
#include "regression_utils.h"

pRegressionParameters initialize_params(size_t num_features) {

	// bias is one term irrespective of features.
	// no need to init using calloc.
	pRegressionParameters params = malloc(sizeof(RegressionParameters));

	if (!params) {
		fprintf(stderr, "parameters (w and b) memory allocation failed\n");
		exit(EXIT_FAILURE);
	}

	float *weights_block = (float *)calloc(num_features, sizeof(float));

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
	float mse; 
	
	float *weights = params->weights;
	float bias = params->bias;
	float **samples = data->samples;
	float *labels = data->labels;
	size_t num_features = data->num_features;

	// Clear weighted errors for each feature
	for (size_t j = 0; j < num_features; j++) {
		diagnostics->weighted_err[j] = 0.0f;
	}

	// For each sample
	for (size_t i = 0; i < data->length; i++) {
		// Compute prediction: y_hat = w1*x1 + w2*x2 + ... + wn*xn + bias
		y_hat = bias;
		for (size_t j = 0; j < num_features; j++) {
			y_hat += weights[j] * samples[j][i];
		}
		
		// Compute error
		err = y_hat - labels[i];
		squared_err += err * err;
		sum_err += err;
		
		// Compute weighted errors for each feature
		for (size_t j = 0; j < num_features; j++) {
			diagnostics->weighted_err[j] += err * samples[j][i];
		}
	}

	mse = squared_err / data->length;

	diagnostics->sum_err = sum_err;
	diagnostics->mse = mse;
}	

void backward_pass(pDataLoader data, pRegressionParameters params, pTrainingDiagnostics diagnostics) {
	size_t num_features = data->num_features;

	// Update weights for each feature
	for (size_t j = 0; j < num_features; j++) {
		float weight_grad = 2.0/data->length * (diagnostics->weighted_err[j]);
		params->weights[j] -= LEARNING_RATE * weight_grad; 
	}

	// Update bias
	float bias_grad = 2.0/data->length * (diagnostics->sum_err);
	params->bias -= LEARNING_RATE * bias_grad;
}

void clear_diagnostics(pTrainingDiagnostics diagnostics, size_t num_features) {
	diagnostics->mse = 0;
	diagnostics->sum_err = 0;
	for (size_t j = 0; j < num_features; j++) {
		diagnostics->weighted_err[j] = 0;
	}
}

int initialize_dataloader(pDataLoader dataloader, size_t num_samples, size_t num_features) {

	assert(num_samples < MAX_SAMPLES);
	assert(num_features < MAX_FEATURES);

	// Allocate array of pointers for each feature
	dataloader->samples = malloc(num_features * sizeof(float*));
	if (!dataloader->samples) {
		fprintf(stderr, "Failed to allocate memory for samples array\n");
		exit(EXIT_FAILURE);
	}

	// Allocate memory for each feature's data
	for (size_t i = 0; i < num_features; i++) {
		dataloader->samples[i] = malloc(num_samples * sizeof(float));
		if (!dataloader->samples[i]) {
			fprintf(stderr, "Failed to allocate memory for feature %zu\n", i);
			exit(EXIT_FAILURE);
		}
	}

	// Allocate memory for labels
	dataloader->labels = malloc(num_samples * sizeof(float));
	if (!dataloader->labels) {
		fprintf(stderr, "Failed to allocate memory for labels\n");
		exit(EXIT_FAILURE);
	}

	dataloader->length = num_samples;
	dataloader->num_features = num_features;

	return EXIT_SUCCESS;
}

void train(pDataLoader data, size_t epochs) {
	pRegressionParameters params = initialize_params(data->num_features);
	pTrainingDiagnostics diagnostics = calloc(1, sizeof(TrainingDiagnostics));
	
	// Allocate memory for weighted errors array
	diagnostics->weighted_err = calloc(data->num_features, sizeof(float));
	if (!diagnostics->weighted_err) {
		fprintf(stderr, "Failed to allocate memory for weighted errors\n");
		exit(EXIT_FAILURE);
	}

	for (size_t i = 0; i < epochs; i++) {
		forward_pass(data, params, diagnostics);
		backward_pass(data, params, diagnostics);
		printf("epoch: %zu - mse: %e\n", i+1, diagnostics->mse);
		printf("weights: ");
		for (size_t j = 0; j < data->num_features; j++) {
			printf("w%zu=%e ", j, params->weights[j]);
		}
		printf("- bias: %e\n", params->bias);
		clear_diagnostics(diagnostics, data->num_features);
	}

	free(params->weights);
	free(params);
	free(diagnostics->weighted_err);
	free(diagnostics);
	
	// Free 2D samples array
	for (size_t i = 0; i < data->num_features; i++) {
		free(data->samples[i]);
	}
	free(data->samples);
	free(data->labels);
	free(data);
}
