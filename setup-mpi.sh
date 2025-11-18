#!/bin/bash
# Setup script for department server - finds and loads MPI module

echo "=== MPI Setup for COP5522-HW3 ==="
echo ""

# Check if MPI is already available
if command -v mpic++ &> /dev/null; then
    echo "✓ MPI compiler already available: $(which mpic++)"
    mpic++ --version | head -n 1
    echo ""
    echo "You can now run: make"
    exit 0
fi

echo "MPI compiler not found in PATH. Searching for modules..."
echo ""

# Try common module names
MPI_MODULES=(
    "mpi/openmpi-x86_64"
    "openmpi"
    "mpi"
    "openmpi/4.1.1"
    "openmpi/4.0.5"
)

for mod in "${MPI_MODULES[@]}"; do
    echo "Trying: module load $mod"
    if module load "$mod" 2>/dev/null; then
        echo "✓ Successfully loaded: $mod"
        if command -v mpic++ &> /dev/null; then
            echo "✓ MPI compiler now available: $(which mpic++)"
            mpic++ --version | head -n 1
            echo ""
            echo "Success! Now run: make"
            echo ""
            echo "To make this permanent, add to your ~/.bashrc:"
            echo "  module load $mod"
            exit 0
        fi
    fi
done

echo ""
echo "❌ Could not automatically load MPI module."
echo ""
echo "Please try manually:"
echo "  1. See available modules: module avail mpi"
echo "  2. Load a module: module load <module-name>"
echo "  3. Then run: make"
echo ""
echo "Or contact your system administrator for MPI installation."
