#include "regression_utils.h"
#include <assert.h>
#include <stddef.h>
#include <stdio.h>
#include <errno.h>
#include <stdlib.h>
#include <string.h>

#define INPUT_FEATURE_INDEX 0
#define OUTPUT_FEATURE_INDEX 1

int csv_parser(char *path, pDataLoader dataloader, size_t nsamples,
               size_t nfeatures) {

  FILE *file;
  file = fopen(path, "r");

  if (!file) {
    fprintf(stderr, "Failed to open file at given path\n");
    exit(EXIT_FAILURE);
  }

  char buffer[256]; // Increased buffer size for multiple features

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
      fprintf(stderr,
              "Invalid CSV format: missing output value at sample %zu\n",
              sample_idx);
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

size_t validate_epochs(char *nepochs_inp) {

	if (strchr(nepochs_inp, '-') != NULL) {
		fprintf(stderr, "Inputted epoch count '%s' cannot be negative\n", nepochs_inp);
		exit(EXIT_FAILURE);
	}

	char *endptr;
	errno = 0;
  size_t num_epochs = (size_t)strtoul(nepochs_inp, &endptr, 10);

	if (endptr == nepochs_inp) {
		fprintf(stderr, "Inputted epoch count '%s' is not a number\n", nepochs_inp);
		exit(EXIT_FAILURE);
	}

	if (*endptr != '\0') {
		fprintf(stderr, "Inputted epoch count '%s' has trailing garbage\n", nepochs_inp);
		exit(EXIT_FAILURE);
	}

	if (errno == ERANGE) {
		fprintf(stderr, "Inputted epoch count '%s' out of range\n", nepochs_inp);
		exit(EXIT_FAILURE);
	}

  if ((num_epochs >= MIN_EPOCHS) && (num_epochs <= MAX_EPOCHS)) {
    printf("Initializing %zu epochs\n", num_epochs);
  } else {
    fprintf(stderr, "Inputted epoch no. invalid\n");
    exit(EXIT_FAILURE);
  }

  return num_epochs;
}

float validate_learning_rate(char *learning_rate_inp) {
	char *endptr;
	errno = 0;
  float learning_rate = strtof(learning_rate_inp, &endptr);

	if (endptr == learning_rate_inp) {
		fprintf(stderr, "Learning rate '%s' is not a number\n", learning_rate_inp);
		exit(EXIT_FAILURE);
	}

	if (*endptr != '\0') {
		fprintf(stderr, "Learning rate '%s' has trailing garbage\n", learning_rate_inp);
		exit(EXIT_FAILURE);
	}

	if (errno == ERANGE) {
		fprintf(stderr, "Learning rate '%s' out of range\n", learning_rate_inp);
		exit(EXIT_FAILURE);
	}

  if ((learning_rate >= MIN_LR) && (learning_rate <= MAX_LR)) {
    printf("Initializing %f learning rate in training configuration\n",
           learning_rate);
  } else {
    fprintf(stderr, "Inputted learning rate invalid\n");
    exit(EXIT_FAILURE);
  }

  return learning_rate;
}

void validate_path(char *buffer, char *path) {

  char *cp;

  cp = path;
  if (*cp == 0) {
    fprintf(stderr, "main: argument an empty string\n");
    exit(1);
  }

  if (!strncpy(buffer, path, PATH_BUF_LEN)) {
    printf("Path validation failed\n");
  }

  printf("%s\n", buffer);
}
