#include <sys/time.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <mpi.h>
#include "PAT_functions.h"

int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);
    
    int rank, P;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &P);
    
    int n = 1000;           // Number of iterations
    int warmup = 100;       // Warm-up iterations
    int chunk_size = 512;
    int buffer_size = 1024;
    
    // Allocate buffers outside the loop
    int *local_data = (int*)malloc(chunk_size * sizeof(int));
    for(int i = 0; i < chunk_size; i++) {
        local_data[i] = rank * 10 + i;
    }
    
    int *result = (int*)malloc(P * chunk_size * sizeof(int));
    
    // Warm-up phase
    for(int i = 0; i < warmup; i++) {
        PAT_AllGather(local_data, chunk_size, result, chunk_size, MPI_COMM_WORLD, buffer_size, chunk_size);
    }
    
    // Synchronize all processes before timing
    MPI_Barrier(MPI_COMM_WORLD);
    
    // Start timing
    double t_start = MPI_Wtime();
    
    for(int i = 0; i < n; i++) {
        PAT_AllGather(local_data, chunk_size, result, chunk_size, MPI_COMM_WORLD, buffer_size, chunk_size);
    }
    
    // End timing
    double t_end = MPI_Wtime();
    double elapsed = t_end - t_start;
    
    // Only rank 0 reports timing
    if(rank == 0) {
        printf("Total time for %d iterations: %f microseconds\n", n, elapsed * 1e6);
        // printf("Time per iteration: %f microseconds\n", (elapsed / n) * 1e6);
    }
    
    free(result);
    free(local_data);
    
    MPI_Finalize();
    return 0;
}
