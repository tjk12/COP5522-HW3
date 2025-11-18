# Makefile for HW3 - MPI Matrix-Vector Multiplication

# Detect OS and allow overrides
UNAME := $(shell uname)

# Allow environment overrides: e.g. `make CXX=mpic++ CXXFLAGS="-O3 -std=c++17"`
CXX ?= mpic++

# Auto-detect MPI compiler wrapper only when CXX was not set externally
ifeq ($(origin CXX), default)
ifneq ($(shell which mpic++ 2>/dev/null),)
CXX := mpic++
else
ifneq ($(shell which mpicxx 2>/dev/null),)
CXX := mpicxx
else
ifneq ($(shell which mpicc 2>/dev/null),)
CXX := mpicc
else
$(info ========================================)
$(info MPI compiler not found in PATH!)
$(info Try one of these solutions:)
$(info   1. Load MPI module: module load mpi/openmpi-x86_64)
$(info   2. Or try: module avail mpi)
$(info   3. Then run 'make' again)
$(info ========================================)
$(error MPI compiler wrapper not found. See instructions above.)
endif
endif
endif
endif

ifeq ($(UNAME), Darwin)
	# macOS: basic optimization (Accelerate removed - use intrinsics instead)
	CXXFLAGS ?= -O3 -std=c++11 -ffast-math -funroll-loops
	MPI_LIBS ?=
else
	# Linux: enable aggressive optimizations with AVX2/FMA for best performance
	CXXFLAGS ?= -O3 -march=native -mavx2 -mfma -std=c++11 -ffast-math -funroll-loops -ftree-vectorize
	MPI_LIBS ?=
endif

# OpenMP compiler for comparison version
OMP_CXX ?= g++
# On macOS, OpenMP support requires libomp; flags differ depending on the OpenMP install.
# If you plan to build the OpenMP variant on macOS, set OMP_FLAGS in the environment, e.g.:
#  make OMP_CXX=clang++ OMP_FLAGS='-O3 -std=c++11 -Xpreprocessor -fopenmp -lomp'
OMP_FLAGS ?= -O3 -march=native -std=c++11 -fopenmp -ffast-math

# Targets
TARGET = hw3
OMP_TARGET = hw3_openmp

# Source files
MPI_SRC = hw3.cpp
OMP_SRC = hw3_openmp.cpp

# Default target
all: $(TARGET)

# MPI version
$(TARGET): $(MPI_SRC)
	$(CXX) $(CXXFLAGS) -o $(TARGET) $(MPI_SRC) $(MPI_LIBS)

# OpenMP version
openmp: $(OMP_TARGET)

$(OMP_TARGET): $(OMP_SRC)
	$(OMP_CXX) $(OMP_FLAGS) -o $(OMP_TARGET) $(OMP_SRC)

# Create sample input file
input.txt:
	echo "1000" > input.txt

# Test targets
test: $(TARGET) input.txt
	mpirun -np 4 ./$(TARGET)

test-openmp: $(OMP_TARGET) input.txt
	./$(OMP_TARGET) 4

# Run performance tests
perf: $(TARGET) input.txt
	@echo "Running performance tests..."
	@for np in 1 2 4 8; do \
		echo "Testing with $$np processes:"; \
		mpirun -np $$np ./$(TARGET); \
	done

# Clean
clean:
	rm -f $(TARGET) $(OMP_TARGET) *.o

# Clean everything including results
cleanall: clean
	rm -f input.txt *.csv *.png

# Help
help:
	@echo "Available targets:"
	@echo "  all        - Build MPI version (default)"
	@echo "  openmp     - Build OpenMP version"
	@echo "  test       - Run quick test with 4 processes"
	@echo "  test-openmp - Run OpenMP test with 4 threads"
	@echo "  perf       - Run basic performance tests"
	@echo "  find-mpi   - Search for available MPI modules"
	@echo "  clean      - Remove executables"
	@echo "  cleanall   - Remove executables and results"
	@echo "  help       - Show this help message"
	@echo ""
	@echo "Example usage:"
	@echo "  make find-mpi          # Find available MPI modules"
	@echo "  module load mpi/...    # Load MPI module"
	@echo "  make                   # Build MPI version"
	@echo "  make test              # Test with 4 processes"
	@echo "  mpirun -np 8 ./hw3     # Run with 8 processes"

# Environment diagnostic target
.PHONY: check-env
check-env:
	@echo "System: $(UNAME)"
	@echo "CXX: $(CXX)"
	@echo "CXXFLAGS: $(CXXFLAGS)"
	@echo "OMP_CXX: $(OMP_CXX)"
	@echo "OMP_FLAGS: $(OMP_FLAGS)"
	@echo "mpirun version:"; mpirun --version || mpirun -h || true

# Helper: detect and show available MPI modules
.PHONY: find-mpi
find-mpi:
	@echo "=== Searching for MPI modules ==="
	@echo "Available MPI modules:"
	@module avail mpi 2>&1 | grep -i mpi || echo "No MPI modules found with 'module avail mpi'"
	@echo ""
	@echo "Available OpenMPI modules:"
	@module avail openmpi 2>&1 | grep -i openmpi || echo "No OpenMPI modules found"
	@echo ""
	@echo "To load a module, run:"
	@echo "  module load <module-name>"
	@echo "  make"

.PHONY: all openmp test test-openmp perf clean cleanall help