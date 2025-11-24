```markdown
# Parallel Aggregated Trees (PAT) AllGather Implementation

## Project Overview

This project implements the Parallel Aggregated Trees (PAT) algorithm for collective MPI communication, specifically for the AllGather operation. PAT is a high-performance collective communication algorithm designed to efficiently gather data from all ranks to all ranks while respecting intermediate buffer size constraints.

## Key Features

- **Logarithmic Phase:** Exponential data aggregation using distance-based partner selection
- **Linear Phase:** Ring-based gathering to collect remaining segments using subgroups
- **Buffer Constraint Support:** Respects maximum intermediate buffer size during communication
- **Non-Power-of-Two Support:** Works correctly with any number of MPI ranks
- **Well-Documented Code:** Comprehensive comments explaining each phase and design decision

## Compilation

### Prerequisites

- **MPI Implementation:** OpenMPI or MPICH
- **C Compiler:** gcc or clang

### Build Commands

```
mpicc -o pat_allgather PAT_AllGatherBase.c -lm
```

## Usage

### Running the Basic Implementation

```
# Run with 8 processes
mpirun -np 8 ./pat_allgather

# Run with 16 processes
mpirun --use-hwthread-cpus -np 16 ./pat_allgather

# Run with non-power-of-two (e.g., 7 ranks)
mpirun -np 7 ./pat_allgather
```

## References

### Academic Publications

- Jeaugey et al. (2012): "PAT: A Parallel Aggregation Tree Algorithm for Scalable Collective Communication"

Relevant for understanding theoretical foundations and performance analysis

### Related MPI Algorithms

- Bruck Algorithm: Traditional approach for AllGather
- Ring AllGather: Simple, scalable baseline
- Recursive Doubling: Fast but requires power-of-two ranks

### MPI Documentation

- Open MPI Documentation
- MPI 3.1 Standard
```
