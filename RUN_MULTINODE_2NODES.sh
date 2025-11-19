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
python3 run_multinode_benchmarks.py

echo ""
echo "Regenerating PDF report with multi-node data..."
python3 report.py

echo ""
echo "=========================================="
echo "Multi-node job complete!"
echo "Check hw3.pdf for complete results"
echo "=========================================="
