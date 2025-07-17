#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <limits.h>
#include "linreg_utils.h"

int main(int argc, char **argv) {
	float x[] = {1, 2, 3, 4, 5};
	float y[] = {3, 5, 7, 9, 11};

	pDataLoader data_loader = malloc(sizeof(DataLoader));
	data_loader->samples = x;
	data_loader->labels = y;
	data_loader->length = 5;

	int num_epochs;

	if (argc > 1) {
		num_epochs = strtol(argv[1], NULL, 10);
	} else {
		fprintf(stderr, "Please indicate number of epochs\n");
		exit(EXIT_FAILURE);
	}

	if ((num_epochs >= MIN_EPOCHS) && (num_epochs <= MAX_EPOCHS)) {
		printf("Initializing %d epochs\n", num_epochs);
	} else {
		fprintf(stderr, "Inputted epoch no. invalid\n");
		exit(EXIT_FAILURE);
	}

	train(data_loader, num_epochs);

	return EXIT_SUCCESS;
}
