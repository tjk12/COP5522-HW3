#!/bin/bash
# RUN_MULTINODE_2NODES.sh - Run multi-node benchmarks with 2 nodes (faster queue)
# Usage: sbatch RUN_MULTINODE_2NODES.sh

#SBATCH --job-name=hw3_2node
#SBATCH --output=hw3_multinode.log
#SBATCH --error=hw3_multinode_error.log
#SBATCH --time=30:00
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=1
#SBATCH --mem=8GB

echo "=========================================="
echo "HW3 Multi-Node Benchmark Run (2 nodes)"
echo "Nodes allocated: $SLURM_JOB_NUM_NODES"
echo "Tasks per node: $SLURM_NTASKS_PER_NODE"
echo "Total tasks: $SLURM_NTASKS"
echo "Date: $(date)"
echo "=========================================="
echo ""

# Load MPI module
echo "Loading MPI module..."
module load openmpi/4.0.5-gcc10.2.0
module list
echo ""

# Build
echo "Building MPI executable..."
make clean
make
echo ""

# Run multi-node benchmarks with 16 processes (2 nodes × 8 tasks)
echo "Running multi-node benchmarks (16 processes across 2 nodes)..."

# Manually run key multi-node tests
echo ""
echo "=== Multi-Node Strong Scaling (16 processes) ==="

for N in 4000 8000; do
    echo "$N" > input.txt
    echo "Testing N=$N, procs=16..."
    mpirun -np 16 ./hw3
done

echo ""
echo "=== Multi-Node Weak Scaling (16 processes) ==="
# N=4000 gives 1M elements per process with 16 procs
echo "4000" > input.txt
echo "Testing N=4000, procs=16 (weak scaling)..."
mpirun -np 16 ./hw3

echo ""
echo "Extending CSV files with multi-node data..."
python3 run_multinode_benchmarks.py

echo ""
echo "Regenerating PDF report with multi-node data..."
python3 report.py

echo ""
echo "=========================================="
echo "Multi-node job complete!"
echo "Check hw3.pdf for complete results"
echo "=========================================="
