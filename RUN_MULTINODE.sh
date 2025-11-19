#!/bin/bash
# RUN_MULTINODE.sh - Run multi-node benchmarks (16-40 processes across multiple nodes)
# Usage: sbatch RUN_MULTINODE.sh

#SBATCH --job-name=hw3_multinode
#SBATCH --output=hw3_multinode.log
#SBATCH --error=hw3_multinode_error.log
#SBATCH --time=60:00
#SBATCH --nodes=5
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=1
#SBATCH --mem=8GB

echo "=========================================="
echo "HW3 Multi-Node Benchmark Run"
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

# Run multi-node benchmarks (extends existing single-node results)
echo "Running multi-node benchmarks (16-40 processes)..."
python3 run_multinode_benchmarks.py

echo ""
echo "Regenerating PDF report with multi-node data..."
python3 report.py

echo ""
echo "=========================================="
echo "Multi-node job complete!"
echo "Check hw3.pdf for complete results with all 4 figures"
echo "=========================================="
