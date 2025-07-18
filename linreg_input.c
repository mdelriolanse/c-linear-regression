#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include "linreg_utils.h"

#define INPUT_FEATURE_INDEX 0
#define OUTPUT_FEATURE_INDEX 1

int csv_parser(char *path, pDataLoader dataloader,
		size_t nsamples, size_t nfeatures) {

	FILE *file;
	printf("About to read file");
	file = fopen(path, "r");
	char buffer[10];

	initialize_dataloader(dataloader, nsamples, nfeatures);

	int sample_idx = 0;
	float output;
	float input;

	while (fgets(buffer, 10, file)) {
		char *input_str = strtok(buffer, ","); // x datapt.
		char *output_str = strtok(NULL, ",");
		
		if ((input_str != NULL) && (output_str != NULL)) {
			output = atof(output_str);
			input = atof(input_str);
		}

		// load data into buffer
		// mechanism will need to be
		// updated for multivariate case
		dataloader->samples[sample_idx] = input;
		dataloader->labels[sample_idx] = output;

		sample_idx++;
	}

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

void validate_path(char *path, char **argv) {

	char *cp;

	cp = *argv;
	if (*cp == 0) {
		fprintf(stderr,"main: argument an empty string\n");
		exit(1);
	}

	if (!strncpy(path, argv[2], PATH_BUF_LEN)) {
		printf("Path validation failed\n");
	}

	printf("%s\n", path);
}
