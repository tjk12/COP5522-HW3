#!/bin/bash
# RUN_COMPLETE.sh - Complete fresh run of all benchmarks
# Usage: sbatch RUN_COMPLETE.sh

#SBATCH --job-name=hw3_complete
#SBATCH --output=hw3_complete.log
#SBATCH --error=hw3_complete_error.log
#SBATCH --time=90:00
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=1
#SBATCH --mem=8GB

echo "=========================================="
echo "HW3 Complete Benchmark Run (Fresh Start)"
echo "Nodes allocated: $SLURM_JOB_NUM_NODES"
echo "Date: $(date)"
echo "=========================================="
echo ""

# Load MPI module
echo "Loading MPI module..."
module load openmpi/4.0.5-gcc10.2.0
module list
echo ""

# Clean everything
echo "Cleaning previous builds and results..."
make clean
rm -f strong_scaling_results.csv weak_scaling_results.csv openmp_results.csv
rm -f *.png hw3.pdf
echo ""

# Build
echo "Building executables..."
make
make openmp
echo ""

# Run OpenMP benchmarks
echo "=========================================="
echo "Step 1: Running OpenMP benchmarks..."
echo "=========================================="
python3 run_openmp_benchmarks.py
echo ""

# Run single-node MPI benchmarks (1-8 processes)
echo "=========================================="
echo "Step 2: Running single-node MPI benchmarks..."
echo "=========================================="
python3 run_benchmarks.py
echo ""

# Run multi-node MPI benchmarks (16, 24, 32 processes)
echo "=========================================="
echo "Step 3: Running multi-node MPI benchmarks..."
echo "=========================================="
python3 run_multinode_benchmarks.py
echo ""

# Generate report
echo "=========================================="
echo "Step 4: Generating PDF report..."
echo "=========================================="
python3 report.py
echo ""

echo "=========================================="
echo "Complete benchmark run finished!"
echo "Generated:"
ls -lh *.csv *.png hw3.pdf 2>/dev/null | awk '{print "  - " $9 " (" $5 ")"}'
echo "=========================================="
