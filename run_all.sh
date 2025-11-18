#!/bin/bash
# run_all.sh - Complete workflow for HW3 benchmarking and report generation
# This script builds both MPI and OpenMP versions, runs all benchmarks, and generates the PDF report

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
YELLOW='\033[1;33m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
MIN_GFLOPS=5.0  # Minimum expected single-core performance
MIN_PARALLEL_GFLOPS=8.0  # Minimum expected parallel performance

echo -e "${BLUE}=================================================="
echo "HW3 Complete Benchmark and Report Pipeline"
echo -e "==================================================${NC}\n"

# Step 1: Clean previous builds
echo -e "${BLUE}[1/6] Cleaning previous builds...${NC}"
make clean 2>/dev/null || true
rm -f strong_scaling_results.csv weak_scaling_results.csv openmp_results.csv
echo -e "${GREEN}✓ Clean complete${NC}\n"

# Step 2: Build MPI version
echo -e "${BLUE}[2/6] Building MPI version (hw3)...${NC}"

# Try to build, if it fails due to missing MPI, attempt to load MPI module
if ! make 2>&1 | tee /tmp/mpi_build.log; then
    if grep -q "MPI compiler wrapper not found" /tmp/mpi_build.log; then
        echo -e "${YELLOW}⚠ MPI compiler not found. Attempting to load MPI module...${NC}"
        
        # Try common MPI module names
        if command -v module &> /dev/null; then
            for mpi_module in mpi/openmpi-x86_64 openmpi mpi/openmpi openmpi-x86_64; do
                echo -e "${BLUE}  Trying: module load $mpi_module${NC}"
                if module load $mpi_module 2>/dev/null; then
                    echo -e "${GREEN}  ✓ Loaded $mpi_module${NC}"
                    # Try building again
                    if make; then
                        echo -e "${GREEN}✓ MPI build successful${NC}\n"
                        break 2
                    fi
                fi
            done
        fi
        
        # If we get here, module loading failed
        echo -e "${RED}✗ Could not load MPI module automatically${NC}"
        echo -e "${YELLOW}Please manually load an MPI module:${NC}"
        echo -e "${YELLOW}  1. Check available modules: module avail mpi${NC}"
        echo -e "${YELLOW}  2. Load a module: module load mpi/openmpi-x86_64${NC}"
        echo -e "${YELLOW}  3. Run this script again${NC}"
        exit 1
    else
        echo -e "${RED}✗ MPI build failed!${NC}"
        cat /tmp/mpi_build.log
        exit 1
    fi
else
    echo -e "${GREEN}✓ MPI build successful${NC}\n"
fi

# Step 3: Build OpenMP version
echo -e "${BLUE}[3/6] Building OpenMP version (hw3_openmp)...${NC}"
if make openmp; then
    echo -e "${GREEN}✓ OpenMP build successful${NC}\n"
else
    echo -e "${RED}✗ OpenMP build failed!${NC}"
    echo -e "${YELLOW}On macOS: brew install libomp${NC}"
    echo -e "${YELLOW}On Linux: install gcc with OpenMP support${NC}"
    exit 1
fi

# Step 4: Run OpenMP benchmarks
echo -e "${BLUE}[4/6] Running OpenMP benchmarks...${NC}"
if python3 run_openmp_benchmarks.py; then
    echo -e "${GREEN}✓ OpenMP benchmarks complete${NC}"
    
    # Check OpenMP results
    if [ -f openmp_results.csv ]; then
        max_gflops=$(awk -F',' 'NR>1 {if ($4 > max) max=$4} END {print max}' openmp_results.csv)
        echo -e "${BLUE}  Peak OpenMP performance: ${max_gflops} Gflop/s${NC}"
        
        if (( $(echo "$max_gflops < $MIN_PARALLEL_GFLOPS" | bc -l) )); then
            echo -e "${YELLOW}  ⚠ Warning: OpenMP performance is below ${MIN_PARALLEL_GFLOPS} Gflop/s${NC}"
            echo -e "${YELLOW}  This may be expected on macOS. Performance should improve on Linux.${NC}"
        else
            echo -e "${GREEN}  ✓ Performance meets expectations${NC}"
        fi
    fi
    echo ""
else
    echo -e "${RED}✗ OpenMP benchmarks failed!${NC}"
    exit 1
fi

# Step 5: Run MPI benchmarks
echo -e "${BLUE}[5/6] Running MPI benchmarks...${NC}"
if python3 run_benchmarks.py; then
    echo -e "${GREEN}✓ MPI benchmarks complete${NC}"
    
    # Check for failures in MPI results
    if grep -q "FAILED" strong_scaling_results.csv 2>/dev/null || \
       grep -q "FAILED" weak_scaling_results.csv 2>/dev/null; then
        echo -e "${YELLOW}  ⚠ Warning: Some MPI tests failed${NC}"
        echo -e "${YELLOW}  This is normal if testing with more processes than available cores.${NC}"
        echo -e "${YELLOW}  On a cluster with more cores, all tests should pass.${NC}"
    fi
    
    # Check MPI single-core performance
    if [ -f strong_scaling_results.csv ]; then
        single_core=$(awk -F',' 'NR>1 && $2==1 {print $4; exit}' strong_scaling_results.csv)
        if [ ! -z "$single_core" ]; then
            echo -e "${BLUE}  Single-core MPI performance: ${single_core} Gflop/s${NC}"
            
            if (( $(echo "$single_core < $MIN_GFLOPS" | bc -l) )); then
                echo -e "${YELLOW}  ⚠ Warning: Single-core performance is below ${MIN_GFLOPS} Gflop/s${NC}"
                echo -e "${YELLOW}  Matrix-vector multiplication is memory-bandwidth bound.${NC}"
                echo -e "${YELLOW}  Performance should improve significantly on Linux server hardware.${NC}"
            else
                echo -e "${GREEN}  ✓ Performance meets expectations${NC}"
            fi
        fi
        
        # Check parallel efficiency
        max_gflops=$(awk -F',' 'NR>1 {if ($4 > max) max=$4} END {print max}' strong_scaling_results.csv)
        echo -e "${BLUE}  Peak MPI performance: ${max_gflops} Gflop/s${NC}"
    fi
    echo ""
else
    echo -e "${RED}✗ MPI benchmarks failed!${NC}"
    exit 1
fi

# Step 6: Generate PDF report
echo -e "${BLUE}[6/6] Generating PDF report...${NC}"
if python3 report.py; then
    echo -e "${GREEN}✓ Report generated successfully${NC}\n"
    
    if [ -f hw3.pdf ]; then
        size=$(ls -lh hw3.pdf | awk '{print $5}')
        echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
        echo -e "${GREEN}SUCCESS! All benchmarks complete.${NC}"
        echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
        echo -e "${BLUE}Generated files:${NC}"
        echo -e "  • hw3.pdf ($size)"
        echo -e "  • strong_scaling_results.csv"
        echo -e "  • weak_scaling_results.csv"
        echo -e "  • openmp_results.csv"
        echo ""
        echo -e "${YELLOW}Next steps:${NC}"
        echo -e "  1. Review hw3.pdf"
        echo -e "  2. Test on Linux cluster for production results"
        echo -e "  3. Run multi-node tests (see WORKFLOW.md)"
        echo ""
    fi
else
    echo -e "${RED}✗ Report generation failed!${NC}"
    echo -e "${YELLOW}Make sure Python dependencies are installed:${NC}"
    
    # Check Python version and provide appropriate install command
    py_version=$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")' 2>/dev/null || echo "3.6")
    if [[ "$py_version" == "3.6" ]]; then
        echo -e "${YELLOW}  For Python 3.6:${NC}"
        echo -e "${YELLOW}  pip3 install --user 'fpdf2<2.6.0' matplotlib pandas numpy${NC}"
    else
        echo -e "${YELLOW}  pip3 install --user fpdf2 matplotlib pandas numpy${NC}"
    fi
    echo -e "${YELLOW}  Or use: pip3 install --user -r requirements.txt${NC}"
    exit 1
fi

echo -e "${BLUE}==================================================${NC}"
echo -e "${GREEN}Pipeline complete!${NC}"
echo -e "${BLUE}==================================================${NC}"
