#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include "regression_utils.h"

#define INPUT_FEATURE_INDEX 0
#define OUTPUT_FEATURE_INDEX 1

int csv_parser(char *path, pDataLoader dataloader,
		size_t nsamples, size_t nfeatures) {

	FILE *file;
	file = fopen(path, "r");

	if (!file) {
		fprintf(stderr, "Failed to open file at given path\n");
		exit(EXIT_FAILURE);
	}

	char buffer[256]; // Increased buffer size for multiple features

	initialize_dataloader(dataloader, nsamples, nfeatures);

	size_t sample_idx = 0;
	float output;
	float *inputs = malloc(nfeatures * sizeof(float));
	if (!inputs) {
		fprintf(stderr, "Failed to allocate memory for input parsing\n");
		exit(EXIT_FAILURE);
	}

	while (fgets(buffer, sizeof(buffer), file) && sample_idx < nsamples) {
		// Parse CSV line: feature1,feature2,...,featureN,output
		char *token = strtok(buffer, ",");
		size_t feature_idx = 0;
		
		// Parse input features
		while (token != NULL && feature_idx < nfeatures) {
			inputs[feature_idx] = atof(token);
			token = strtok(NULL, ",");
			feature_idx++;
		}
		
		// Parse output (last token)
		if (token != NULL) {
			output = atof(token);
		} else {
			fprintf(stderr, "Invalid CSV format: missing output value at sample %zu\n", sample_idx);
			exit(EXIT_FAILURE);
		}

		// Store data in 2D array format: samples[feature][sample_index]
		for (size_t j = 0; j < nfeatures; j++) {
			dataloader->samples[j][sample_idx] = inputs[j];
		}
		dataloader->labels[sample_idx] = output;

		sample_idx++;
	}

	free(inputs);
	fclose(file);
	return EXIT_SUCCESS;
}

size_t validate_epochs(char *nepochs) {
	size_t num_epochs = (size_t) atoi(nepochs);

	if ((num_epochs >= MIN_EPOCHS) && (num_epochs <= MAX_EPOCHS)) {
		printf("Initializing %zu epochs\n", num_epochs);
	} else {
		fprintf(stderr, "Inputted epoch no. invalid\n");
		exit(EXIT_FAILURE);
	}

	return num_epochs;
}

void validate_path(char *buffer, char *path) {

	char *cp;

	cp = path;
	if (*cp == 0) {
		fprintf(stderr,"main: argument an empty string\n");
		exit(1);
	}

	if (!strncpy(buffer, path, PATH_BUF_LEN)) {
		printf("Path validation failed\n");
	}

	printf("%s\n", buffer);
}
