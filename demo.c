#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <limits.h>
#include "linreg_utils.h"

int main(int argc, char **argv) {

	pDataLoader data_loader = malloc(sizeof(DataLoader));

	int num_epochs;


	if (argc > 1) {
		validate_epochs(argv[1]);
	} else {
		fprintf(stderr, "Please indicate number of epochs\n");
		exit(EXIT_FAILURE);
	}

	valid


	train(data_loader, num_epochs);

	return EXIT_SUCCESS;
}
