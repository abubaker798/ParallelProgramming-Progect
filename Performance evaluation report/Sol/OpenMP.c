#include <stdlib.h> 
#include <stdio.h>  
#include <omp.h>    
#include <math.h>
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

#define THREADS_NUMBER 5
#define ARRAY_SIZE 16384

int main(int argc, char *argv[])
{
    omp_set_num_threads(THREADS_NUMBER);
    int n = ARRAY_SIZE;
    int data[ARRAY_SIZE]; // Array holds the elements
    double start, end;
    double mean = 0.0;

    double variance = 0.0, standard_deviation;
    start = omp_get_wtime();

    // Initialize data with values
#pragma omp parallel for
    for (int i = 0; i < n; i++)
    {
        data[i] = i + 1;
    }

    // Calculate mean
#pragma omp parallel for reduction(+ : mean)
    for (int i = 0; i < n; i++)
    {
        mean += data[i];
    }
    mean /= n;

// Calculate variance
#pragma omp parallel for reduction(+ : variance)
    for (int i = 0; i < n; i++)
    {
        variance += ((data[i] - mean) * (data[i] - mean));
    }

    variance /= n;
    standard_deviation = sqrt(variance);
    end = omp_get_wtime();

    printf("Mean = %f\n", mean);
    printf("Variance = %f\n", variance);
    printf("Standard deviation = %f\n", standard_deviation);
    printf("Elapsed time is %f in Seconds.\n", end - start);
    return 0;
}
