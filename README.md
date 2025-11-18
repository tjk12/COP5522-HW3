# COP5522-HW3 - MPI Matrix-Vector Multiplication

## Overview
High-performance MPI implementation of matrix-vector multiplication optimized for Linux x86-64 systems with AVX2/FMA support.

## Linux Optimization Features
This code is optimized for Linux HPC environments:
- **AVX2/FMA intrinsics**: Primary code path uses 256-bit SIMD with fused multiply-add
- **Aggressive compiler flags**: `-march=native -mavx2 -mfma -ftree-vectorize`
- **Cache-friendly layout**: Row-major storage with optimized memory access patterns
- **Loop unrolling**: 16-element unrolling in AVX2 path for better ILP

## Building on Linux HPC Clusters

### Quick Setup (Department Server)
If you get an MPI compiler error, run the setup script:
```bash
source setup-mpi.sh
make
```

Or manually find and load MPI:
```bash
make find-mpi          # See available MPI modules
module load mpi/openmpi-x86_64  # Load the module
make                   # Build
```

### Basic Build
```bash
make
```

### On SLURM/PBS Clusters
If MPI is provided via modules:
```bash
module load openmpi  # or module load mpi/openmpi-x86_64
make
```

### Performance Build (Linux)
The Makefile automatically applies these flags on Linux:
```bash
-O3 -march=native -mavx2 -mfma -ffast-math -funroll-loops -ftree-vectorize
```

### Verifying AVX2 Support
Check if your CPU supports AVX2:
```bash
lscpu | grep -i avx2
# or
cat /proc/cpuinfo | grep avx2
```

## Running

### Single Node
```bash
mpirun -np 4 ./hw3        # 4 processes
mpirun -np 8 ./hw3        # 8 processes
```

### On SLURM Clusters
```bash
srun -n 16 ./hw3          # 16 processes
```

### On PBS Clusters
```bash
mpiexec -n 16 ./hw3       # 16 processes
```

## Performance Expectations (Linux x86-64 with AVX2)

On modern Intel/AMD processors (Skylake+, Zen2+):
- **Single core**: 5-10 Gflop/s (memory bandwidth limited)
- **4 cores**: 15-30 Gflop/s (3-4x speedup)
- **8 cores**: 25-45 Gflop/s (diminishing returns from memory bandwidth)

Matrix-vector multiplication is memory-bound, so performance scales with memory bandwidth rather than compute capacity.

## Cross-Platform Notes

### macOS
The code compiles on macOS but without AVX2 optimizations (Apple Silicon uses ARM NEON). Performance will be lower than Linux x86-64.

### ARM/NEON Systems
ARM-specific optimizations using NEON intrinsics are included but less mature than the AVX2 path.

## Performance Testing

Run the full benchmark suite:
```bash
python3 performance_report.py
```

This generates:
- `strong_scaling_results.csv` - Fixed problem size, varying processes
- `weak_scaling_results.csv` - Constant work per process

Generate the PDF report:
```bash
python3 report.py
```

## Code Structure

- `hw3.cpp` - Main MPI implementation with AVX2 optimizations
- `Makefile` - Cross-platform build system (Linux-optimized)
- `report.py` - PDF report generation (requires fpdf2, matplotlib)
- `performance_report.py` - Automated benchmarking script
- `input.txt` - Matrix size input (single integer N for NxN matrix)

## Optimization Details

The code uses a three-tier approach:
1. **AVX2 path** (Linux x86-64): 256-bit SIMD with FMA, 16-element unrolling
2. **NEON path** (ARM): 128-bit SIMD with multiply-accumulate
3. **Scalar path** (fallback): Manual 8-way unrolling

On Linux systems with AVX2, the compiler will use path #1, achieving 3-5x better single-core performance than the scalar path.