#!/bin/bash
# RUN_ON_CLUSTER.sh - Simple script to run everything on the cluster
# Usage: sbatch RUN_ON_CLUSTER.sh

#SBATCH --job-name=hw3_bench
#SBATCH --output=hw3_output.log
#SBATCH --error=hw3_error.log
#SBATCH --time=30:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=1
#SBATCH --mem=8GB

echo "=========================================="
echo "HW3 Automated Benchmark Run"
echo "Node: $(hostname)"
echo "Date: $(date)"
echo "=========================================="
echo ""

# Load MPI module
echo "Loading MPI module..."
module load openmpi/4.0.5-gcc10.2.0
module list
echo ""

# Install Python dependencies if needed
echo "Checking Python dependencies..."
if ! python3 -c "import fpdf" 2>/dev/null; then
    echo "Installing Python packages..."
    pip3 install --user 'numpy<1.20' 'pandas<1.2' 'matplotlib<3.4' 'fpdf2<2.6.0'
    echo ""
fi

# Run the complete pipeline
echo "Running complete benchmark pipeline..."
./run_all.sh

echo ""
echo "=========================================="
echo "Job complete!"
echo "Check hw3.pdf for results"
echo "=========================================="
