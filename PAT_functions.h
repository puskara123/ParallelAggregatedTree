#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <mpi.h>

int calculate_log_steps(int P, int buffer_size);
void print_array(int rank, const char* label, int* arr, int size);
void reorder(int* arr, int size, int rank, int chunk_size);
int pcount(int rank, int index, int total);
void Ring_AllGather(int rank, int* sendbuf, int sendcount, int* recvbuf, int recvcount, MPI_Comm comm, int P, int * buf_size, int index, int chunk_size);
void PAT_AllGather(int* sendbuf, int sendcount, int* recvbuf, int recvcount, MPI_Comm comm, int buf_size, int chunk_size);