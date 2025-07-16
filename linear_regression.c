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

int main(int argc, char **argv) {
	float x[] = {1, 2, 3, 4, 5};
	float y[] = {3, 5, 7, 9, 11};

	pDataLoader data_loader = malloc(sizeof(DataLoader));
	data_loader->samples = x;
	data_loader->labels = y;
	data_loader->length = 5;
	int num_epochs = strtol(argv[1], NULL, 10);

	if (num_epochs == LONG_MIN) || (num_epochs == LONG_MAX) {
		fprintf(stderr, "Inputted epoch no. invalid");
	}

	train(data_loader, num_epochs);

	return EXIT_SUCCESS;
}

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

// g
