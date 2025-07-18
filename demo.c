#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <limits.h>
#include "linreg_utils.h"

int main(int argc, char **argv) {

	pDataLoader data_loader = malloc(sizeof(DataLoader));

	size_t num_epochs;
	char pathbuf[PATH_BUF_LEN];

	if (argc > 1) {
		num_epochs = validate_epochs(argv[1]);
	} else {
		fprintf(stderr, "Please indicate number of epochs\n");
		exit(EXIT_FAILURE);
	}

	validate_path(pathbuf, argv);

	initialize_dataloader(data_loader, 10, 1);

	csv_parser(pathbuf, data_loader, 10, 1);

	train(data_loader, num_epochs);

	return EXIT_SUCCESS;
}
