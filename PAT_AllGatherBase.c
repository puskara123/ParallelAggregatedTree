#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <mpi.h>

// Calculate number of logarithmic steps possible for any P and the buffer size
int calculate_log_steps(int P, int buffer_size){
    int m = P < buffer_size ? P : buffer_size;
    int steps = 0;
    int temp = 1;
    while(temp < m){
        temp *= 2;
        steps ++;
    }
    return steps;
}

// Print array for debugging
void print_array(int rank, const char* label, int* arr, int size) {
    printf("Rank %d %s: [", rank, label);
    for (int i = 0; i < size; i++) {
        printf("%d", arr[i]);
        if (i < size - 1) printf(", ");
    }
    printf("]\n");
    fflush(stdout);
}

// Reorder the array
void reorder(int* arr, int size, int rank, int chunk_size){
    int *buffer = (int*)malloc(size * sizeof(int));
    memcpy(buffer, arr, size * sizeof(int));
    int P = (size / chunk_size);
    for(int i = 0; i < P; i ++){
        for(int j = 0; j < chunk_size; j ++){
            // arr[size - ((i + size - rank - 1) % size) - 1] = buffer[i];
            arr[(P - ((i + P - rank - 1) % P) - 1) * chunk_size + j] = buffer[i * chunk_size + j];
            //printf("Old : %d, New : %d\n", i, (P - ((i + P - rank - 1) % P) - 1));
        }
    }
    free(buffer);
}

// get the number of steps in the linear phase
int pcount(int rank, int index, int total){
    return (1 << (int)ceil(log2(total)) - (int)(log2(index)));
}

// Modified Ring AllGather Implementation
void Ring_AllGather(int rank, int* sendbuf, int sendcount, int* recvbuf, int recvcount, MPI_Comm comm, int P, int * buf_size, int index, int chunk_size){
    printf("\n=== Rank %d starting Linear Phase ===\n", rank);
    int size = pcount(rank, index, P);
    // printf("%d: rank, %d: size, %d: index, %d: P, %d: chunk, %d: sc\n", rank, size, index, P, chunk_size, sendcount);
    int offset = sendcount * chunk_size;
    
    for(int i = 1; i < size; i ++){
        
        // calculate partner/ size of communication
        printf("rank: %d, sendcount: %d, recvcount: %d, buf_size: %d\n", rank, sendcount, recvcount, (*buf_size));
        // printf("rank: %d, sendcount: %d, recvcount: %d, buf_size: %d, index: %d\n", rank, sendcount, recvcount, (*buf_size), index);
        sendcount = ((sendcount * chunk_size + (*buf_size)) >= P * chunk_size) ? ((P * chunk_size - (*buf_size)) / chunk_size) : sendcount;
        recvcount = ((recvcount * chunk_size + (*buf_size)) >= P * chunk_size) ? ((P * chunk_size - (*buf_size)) / chunk_size) : recvcount;
        // printf("%d: rank, %d: size, %d: index, %d: P, %d: chunk, %d: sc\n", rank, size, index, P, chunk_size, sendcount);
        int send_index = (i * index + rank) % P;
        int recv_index = (rank - (i * index) + (P * i)) % P;
        // printf("%d: send, %d: receive\n", send_index, recv_index);
        // printf("rank: %d, sendcount: %d, recvcount: %d, buf_size: %d\n", rank, sendcount, recvcount, (*buf_size));
        
        // Exchange data with partner using non-blocking communication
        MPI_Request requests[2];
        MPI_Status statuses[2];
        
        MPI_Isend(sendbuf, sendcount * chunk_size, MPI_INT, send_index, i, comm, &requests[0]);
        MPI_Irecv(recvbuf, recvcount * chunk_size, MPI_INT, recv_index, i, comm, &requests[1]);

        (*buf_size) += (sendcount * chunk_size);
        
        // Wait for both operations to complete
        MPI_Waitall(2, requests, statuses);

        //print_array(rank, "Received", recvbuf, (recvcount * chunk_size));

        memcpy(sendbuf + i * offset, recvbuf, sendcount * sizeof(int) * chunk_size);
        print_array(rank, "Buffer After Step", sendbuf, (*buf_size));
    }
    MPI_Barrier(comm);
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////

// PAT AllGather implementation with non-power-of-two support
void PAT_AllGather(int* sendbuf, int sendcount, int* recvbuf, int recvcount, MPI_Comm comm, int buf_size, int chunk_size) {
    int rank, P;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &P);
    
    printf("\n=== Rank %d starting PAT AllGather ===\n", rank);
    fflush(stdout);
    
    // Step 1: Initialize local buffer with own data
    int* buffer = (int*)malloc(P * sendcount * sizeof(int));
    memcpy(buffer, sendbuf, sendcount * sizeof(int));
    int buffer_size = sendcount;  // Current number of elements in buffer
    
    print_array(rank, "Initial buffer", buffer, buffer_size);
    
    // Step 2: Calculate number of logarithmic steps
    int log_steps = calculate_log_steps(P, (int)(buf_size / chunk_size));
    // printf("Rank %d: Will perform %d logarithmic steps (P=%d)\n", rank, log_steps, P);
    fflush(stdout);
    
    // Step 3: Logarithmic phase
    for (int step = 0; step < log_steps; step++) {
        int distance = 1 << step;  // 2^step
        
        // Calculate communication partners using modulo
        int send_partner = (rank + distance) % P;
        int recv_partner = (rank - distance + P) % P;
        
        printf("Rank %d, Step %d: distance=%d, sending to %d, receiving from %d\n", rank, step, distance, send_partner, recv_partner);
        fflush(stdout);
        
        // Allocate send and receive buffers
        int* send_buf = (int*)malloc(buffer_size * sizeof(int));
        int* recv_buf = (int*)malloc(P * sendcount * sizeof(int));
        
        // Copy current buffer to send buffer
        memcpy(send_buf, buffer, buffer_size * sizeof(int));
        
        // Exchange data with partner using non-blocking communication
        MPI_Request requests[2];
        MPI_Status statuses[2];
        
        MPI_Isend(send_buf, buffer_size, MPI_INT, send_partner, step, comm, &requests[0]);
        MPI_Irecv(recv_buf, P * sendcount, MPI_INT, recv_partner, step, comm, &requests[1]);
        
        // Wait for both operations to complete
        MPI_Waitall(2, requests, statuses);
        
        // Get actual received count
        int recv_count;
        MPI_Get_count(&statuses[1], MPI_INT, &recv_count);
        
        // printf("Rank %d, Step %d: sent %d elements, received %d elements\n", rank, step, buffer_size, recv_count);
        fflush(stdout);
        
        // Merge received data into buffer (append at end)
        if (buffer_size + recv_count <= P * sendcount) {
            memcpy(buffer + buffer_size, recv_buf, recv_count * sizeof(int));
            buffer_size += recv_count;
        } else {
            // printf("Rank %d: Buffer overflow at step %d! Capping buffer size.\n", rank, step);
            fflush(stdout);
            // Cap at maximum
            int remaining = P * sendcount - buffer_size;
            if (remaining > 0) {
                memcpy(buffer + buffer_size, recv_buf, remaining * sizeof(int));
                buffer_size = P * sendcount;
            }
        }
        
        print_array(rank, "Buffer After Step", buffer, buffer_size);
        
        free(send_buf);
        free(recv_buf);
        
        // Barrier to ensure all processes are synchronized
        MPI_Barrier(comm);
    }
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////

    // Step 4: Calculate parameters and start linear phase
    int index = (2 << (log_steps - 1));
    int * recvbuffer = (int*)malloc(P * sendcount * sizeof(int));
    Ring_AllGather(rank, buffer, index, recvbuffer, index, comm, P, &buffer_size, index, chunk_size);

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    
    // Step 5: Copy final buffer to output
    reorder(buffer, buffer_size, rank, chunk_size);
    int copy_size = (buffer_size < P * recvcount) ? buffer_size : P * recvcount;
    memcpy(recvbuf, buffer, copy_size * sizeof(int));
    
    printf("Rank %d: Final buffer size = %d\n", rank, buffer_size);
    print_array(rank, "Final output", recvbuf, P * recvcount);
    fflush(stdout);
    
    free(buffer);
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    
    int rank, P;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &P);
    
    printf("Process %d of %d started\n", rank, P);
    fflush(stdout);

///////////////////////////////////////////////////////////////////////////////////////////////////////////////    
    // Each process has one integer value (its rank * 10)
    int chunk_size = 2;
    int *local_data = (int*)malloc(chunk_size * sizeof(int));
    for(int i = 0; i < chunk_size; i ++){
        local_data[i] = rank * 10 + i;
    }
///////////////////////////////////////////////////////////////////////////////////////////////////////////////

    int* result = (int*)malloc(P * chunk_size * sizeof(int));
    
    // Perform PAT AllGather 
    PAT_AllGather(local_data, chunk_size, result, chunk_size, MPI_COMM_WORLD, 4, chunk_size);
    
    // Verify results
    // printf("\n=== Rank %d FINAL RESULT ===\n", rank);
    // print_array(rank, "Gathered data", result, P);
    
    // Compare with standard MPI_Allgather for verification
    int* expected = (int*)malloc(P * chunk_size * sizeof(int));
    MPI_Allgather(local_data, chunk_size, MPI_INT, expected, chunk_size, MPI_INT, MPI_COMM_WORLD);
    
    /* printf("Rank %d: Expected result: [", rank);
    for (int i = 0; i < P; i++) {
        printf("%d", expected[i]);
        if (i < P - 1) printf(", ");
    }
    printf("]\n"); */
    fflush(stdout);

    // Checking Correctness of Output
    int flag = 1;
    for(int i = 0; i < P; i ++){
        if(expected[i] != result[i]){
            flag = 0;
            break;
        }
    }
    if(flag){
        printf("=== Rank %d has correct output ===\n", rank);
    }
    else{
        printf("XXX Rank %d has INCORRECT output XXX\n", rank);
    }
    
    free(result);
    free(expected);
    free(local_data);
    
    MPI_Finalize();
    return 0;
}