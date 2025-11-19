#!/bin/bash
# RUN_MULTINODE_SEQUENTIAL.sh - Run 2, 3, and 4 node tests sequentially
# Usage: sbatch RUN_MULTINODE_SEQUENTIAL.sh

#SBATCH --job-name=hw3_multi
#SBATCH --output=hw3_multinode.log
#SBATCH --error=hw3_multinode_error.log
#SBATCH --time=90:00
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=1
#SBATCH --mem=8GB

echo "=========================================="
echo "HW3 Multi-Node Sequential Benchmark"
echo "Will test 16, 24, 32 processes"
echo "Nodes allocated: $SLURM_JOB_NUM_NODES"
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

# Run benchmarks with 16, 24, 32 processes
echo "Running multi-node benchmarks..."
python3 run_multinode_benchmarks.py

echo ""
echo "Regenerating PDF report..."
python3 report.py

echo ""
echo "=========================================="
echo "All multi-node tests complete!"
echo "Tested: 16, 24, 32 processes"
echo "Check hw3.pdf for complete results"
echo "=========================================="
