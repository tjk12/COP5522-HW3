#!/bin/bash
# QUICK_TEST.sh - Quick interactive test (5 minutes)
# Usage: ./QUICK_TEST.sh

echo "=========================================="
echo "Starting interactive session..."
echo "=========================================="
echo ""

interact -t 5:00 -N 1 --ntasks-per-node=8 << 'EOF'
# Load MPI
module load openmpi/4.0.5-gcc10.2.0

# Run benchmarks
./run_all.sh

# Done
echo ""
echo "Test complete! Check hw3.pdf"
exit
EOF
