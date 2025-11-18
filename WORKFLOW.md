# HW3 Workflow - Complete Guide

## Quick Start (Department Server / Bridges-2)

### 1. Setup MPI
```bash
# On department server
source setup-mpi.sh
# OR manually:
module load mpi/openmpi-x86_64

# On Bridges-2
module load openmpi/4.0.5-gcc10.2.0
```

### 2. Build
```bash
make
```

### 3. Run Experiments
```bash
# Generate CSV data files with benchmark results
python3 run_benchmarks.py
```

This will create:
- `strong_scaling_results.csv` - Fixed problem size, varying processes
- `weak_scaling_results.csv` - Constant work per process

### 4. Generate PDF Report
```bash
python3 report.py
```

This creates `hw3.pdf` with all required figures and analysis.

## What the Assignment Requires

### Code Requirements ✓
- [x] `hw3.cpp` - MPI matrix-vector multiplication
- [x] Process 0 reads matrix size from `input.txt`
- [x] Point-to-point communication (sends/receives) for broadcasting size
- [x] User-defined datatype (MPI_Type_contiguous)
- [x] Collective communication (MPI_Gatherv) for gathering results
- [x] Efficient memory usage (only local rows stored per process)
- [x] Command-line argument for number of processes
- [x] Timing excludes I/O operations

### Report Requirements
Your `hw3.pdf` should include:

1. **Four figures:**
   - Strong scaling on single node (speedup vs cores)
   - Strong scaling on multiple nodes (speedup vs nodes)
   - Weak scaling on single node (speedup vs cores)
   - Weak scaling on multiple nodes (speedup vs nodes)

2. **Comparison table:**
   - MPI vs OpenMP performance (Gflop/s) for different core counts on one node

3. **Conclusions:**
   - For what matrix sizes is parallelism worthwhile?
   - How does speedup change on multiple nodes?
   - Weak scaling and strong scaling analysis

4. **AI tool reflection:**
   - How AI tools helped with MPI code, debugging, optimization
   - What they were useful for
   - Where they fell short
   - Impact on your programming role

## Running on Bridges-2 (Required for Final Submission)

### Single Node Tests
```bash
# Interactive session
salloc -N 1 -n 4 -p RM-shared -t 1:00:00

# Run benchmarks
python3 run_benchmarks.py
```

### Multi-Node Tests (Required)
You need to test on 2-5 nodes. Create a SLURM script:

```bash
#!/bin/bash
#SBATCH -N 4              # Number of nodes
#SBATCH -n 16             # Total number of processes (4 per node)
#SBATCH -p RM             # Partition
#SBATCH -t 1:00:00        # Time limit

module load openmpi/4.0.5-gcc10.2.0

# Test different configurations
for nodes in 1 2 4 5; do
    procs=$((nodes * 4))  # 4 cores per node (optimal from single-node tests)
    echo "Testing with $nodes nodes, $procs processes"
    mpirun -np $procs ./hw3
done
```

## Current Experiment Configuration

The `run_benchmarks.py` script runs:

**Strong Scaling:**
- Matrix sizes: N = 1000, 2000, 4000, 8000
- Process counts: 1, 2, 4 (configurable)

**Weak Scaling:**
- Base size: N = 1000
- Process counts: 1, 2, 4
- Total N scales as: N_total = N_base × sqrt(processes)

## Customizing Experiments

Edit `run_benchmarks.py` to change test configurations:

```python
# For more thorough testing
strong_results = runner.run_strong_scaling(
    matrix_sizes=[1000, 2000, 4000, 8000, 16000],
    process_counts=[1, 2, 4, 8, 16, 32]
)

# For multi-node tests
weak_results = runner.run_weak_scaling(
    base_sizes=[1000, 2000],
    process_counts=[1, 2, 4, 8, 16, 32, 64]
)
```

## Manual Testing

Test individual configurations:
```bash
# Single test
echo "8000" > input.txt
mpirun -np 4 ./hw3

# Or specify expected process count as argument
mpirun -np 4 ./hw3 4
```

## OpenMP Comparison (TODO)

The assignment requires comparing MPI vs OpenMP. You need:

1. An OpenMP implementation (`hw3_openmp.cpp`)
2. Run comparable tests with OpenMP
3. Add comparison table to report

To build OpenMP version:
```bash
make openmp
OMP_NUM_THREADS=4 ./hw3_openmp
```

## File Structure for Submission

Your `hw3` directory should contain:
```
hw3/
├── hw3.cpp           # MPI implementation
├── hw3.pdf           # Performance report
├── LOG.txt           # Build/run log
├── input.txt         # Matrix size input
├── Makefile          # Build system
└── (other supporting files)
```

## Submission Checklist

- [ ] Code compiles with `make` on cs-ssh
- [ ] `./hw3` runs with one command-line argument
- [ ] `hw3.pdf` includes all 4 required figures
- [ ] Report has comparison table (MPI vs OpenMP)
- [ ] Report has conclusions about when parallelism is worthwhile
- [ ] Report discusses multi-node scaling
- [ ] Report includes AI tool reflection
- [ ] All tests run on Bridges-2 (required for performance numbers)
- [ ] Single-core performance > 10 Gflop/s (or explained if not)

## Troubleshooting

**"No MPI compiler found"**
```bash
make find-mpi
module load mpi/openmpi-x86_64
```

**"mpirun not found"**
```bash
module load openmpi
```

**Python dependencies missing**
```bash
pip install fpdf2 matplotlib pandas numpy
```

**Low performance**
- Check if AVX2 is enabled: `lscpu | grep avx2`
- Verify compilation flags: `make check-env`
- Ensure running on compute nodes, not login nodes
