#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include "linreg_utils.h"

#define INPUT_FEATURE_INDEX 0
#define OUTPUT_FEATURE_INDEX 1

void csv_parser(char *path, pDataLoader dataloader,
		size_t nsamples, size_t nfeatures) {

	FILE *file;
	file = fopen(path, "r");
	char buffer[10];

	initialize_dataloader(dataloader, nsamples, nfeatures);

	int sample_idx = 0;

	while (fgets(buffer, 10, file)) {
		char *input_str = strtok(buffer, ","); // x datapt.
		float input = atof(input_str);
		
		char *output_str = strtok(NULL, ",");
		float output = atof(output_str);

		// load data into buffer
		// mechanism will need to be
		// updated for multivariate case
		dataloader->samples[sample_idx] = input;
		dataloader->labels[sample_idx] = output;

		sample_idx++;
	}
}

void validate_epochs(char *nepochs) {
	int num_epochs;

	if ((num_epochs >= MIN_EPOCHS) && (num_epochs <= MAX_EPOCHS)) {
		printf("Initializing %d epochs\n", num_epochs);
	} else {
		fprintf(stderr, "Inputted epoch no. invalid\n");
		exit(EXIT_FAILURE);
	}
}

char *validate_path(char *path, char **argv) {

	char *cp;

	cp = *argv;
	if (*cp == 0) {
		fprintf(stderr,"main: argument an empty string\n");
		exit(1);
	}

	if (!strncpy(path, argv[1], sizeof(*path))) {
		printf("Path validation failed\n");
	}

	puts(path);

	return path;
}
