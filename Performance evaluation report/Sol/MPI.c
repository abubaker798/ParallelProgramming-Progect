#include "mpi.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

/*
 *  In MPI programs, the main function for the program is run on every
 *  process that gets initialized when you start up this code using mpirun.
 */

// Each process calculates its local sum on its portion of data.
// If the division result has a remainder, then send this remainder to the last process
// (the process that has the rank "n-1" or let the master process work on it).

// The master process calculates the mean(dividing the total sum by the size of the elements)
// and sends it to all processes.

// Each process calculates the squared difference on its portion of data.

// The master process then calculates the variance
// (dividing the total squared difference by the size of the elements).

// The master process calculates the standard deviation
// by getting the square root of the variance and prints the results..

#define MASTER 0 // One process will take care of initialization
#define ARRAY_SIZE 16384

int main(int argc, char *argv[])
{
	int p;		 // Total number of processes
	int my_rank; // Rank of each process

	// Initialization of MPI environment
	MPI_Init(&argc, &argv);
	MPI_Comm_size(MPI_COMM_WORLD, &p);
	// Now you know the total number of processes running in parallel
	MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
	// Now you know the rank of the current process

	// Shared variables
	int n = ARRAY_SIZE; // Array size
	int data[n];		// Array holds the elements

	// Local variables
	int my_n = n / p;
	int *my_data = (int *)malloc(my_n * (sizeof(int)));
	double my_mean = 0.0;
	double my_deviations = 0.0;

	// Master variables
	double start, end;
	double mean = 0.0, variance, standard_deviation;
	double deviations = 0.0;

	// We choose process rank 0 to be the root, or master,
	// Which will be used to  initialize the full arrays.
	if (my_rank == MASTER)
	{
		// Initialize data with values
		for (int i = 0; i < n; i++)
		{
			data[i] = i + 1;
		}

		start = MPI_Wtime(); // Elapsed time in seconds
	}

	// Scatter array to all slaves
	MPI_Scatter(data, my_n, MPI_INT, my_data, my_n, MPI_INT, MASTER, MPI_COMM_WORLD);

	// Calculate my_mean
	for (int i = 0; i < my_n; i++)
	{
		my_mean += my_data[i];
	}

	// Handling reminder
	if (my_rank == MASTER)
	{
		for (int i = (my_n * p); i < n; i++)
		{
			my_mean += data[i];
		}
	}

	MPI_Allreduce(&my_mean, &mean, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

	mean /= n;

	// Calculate deviations
	for (int i = 0; i < my_n; i++)
	{
		my_deviations += ((my_data[i] - mean) * (my_data[i] - mean));
	}

	if (my_rank == MASTER)
	{
		for (int i = (my_n * p); i < n; i++)
		{
			my_deviations += ((data[i] - mean) * (data[i] - mean));
		}
	}

	MPI_Reduce(&my_deviations, &deviations, 1, MPI_DOUBLE, MPI_SUM, MASTER, MPI_COMM_WORLD);

	if (my_rank == MASTER)
	{
		variance = deviations / n;
		standard_deviation = sqrt(variance);
		end = MPI_Wtime();
		printf("Mean = %f\n", mean);
		printf("Variance = %f\n", variance);
		printf("Standard deviation = %f\n", standard_deviation);
		printf("Elapsed time is %f in Seconds.\n", end - start);
	}

	// clean up memory
	if (my_rank != MASTER)
	{
		free(my_data);
	}

	// Terminate MPI Environment and Processes
	MPI_Finalize();

	return 0;
}