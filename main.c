#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <limits.h>
#include "regression_utils.h"

int main(int argc, char **argv) {

	if (argc != 5) {
		printf("Usage: %s <num_epochs> <num_features> <num_samples> <data_file_path>\n", argv[0]);
		printf("Example: %s 1000 2 100 sample_data.csv\n", argv[0]);
		exit(EXIT_FAILURE);
	}

	pDataLoader data_loader = malloc(sizeof(DataLoader));

	size_t num_epochs;
	size_t num_features;
	size_t num_samples;
	char pathbuf[PATH_BUF_LEN];

	// Parse number of epochs
	num_epochs = validate_epochs(argv[1]);
	
	// Parse number of features
	num_features = (size_t)atoi(argv[2]);
	if (num_features < 1 || num_features > MAX_FEATURES) {
		fprintf(stderr, "Number of features must be between 1 and %d\n", MAX_FEATURES);
		exit(EXIT_FAILURE);
	}
	printf("Using %zu features\n", num_features);

	// Parse number of samples
	num_samples = (size_t)atoi(argv[3]);
	if (num_samples < 1 || num_samples > MAX_SAMPLES) {
		fprintf(stderr, "Number of samples must be between 1 and %d\n", MAX_SAMPLES);
		exit(EXIT_FAILURE);
	}
	printf("Using %zu samples\n", num_samples);

	// Validate and set data file path
	validate_path(pathbuf, argv[4]);

	// Parse CSV data
	csv_parser(pathbuf, data_loader, num_samples, num_features);

	train(data_loader, num_epochs);

	return EXIT_SUCCESS;
}
