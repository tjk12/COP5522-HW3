# Quick Start for Department Server

## If you get "No MPI compiler wrapper found" error:

### Option 1: Auto-setup (Easiest)
```bash
source setup-mpi.sh
make
```

### Option 2: Manual setup
```bash
# Find available MPI modules
make find-mpi
# OR
module avail mpi

# Load the MPI module (common names):
module load mpi/openmpi-x86_64
# OR
module load openmpi

# Then build
make
```

### Option 3: Make it permanent
Add to your `~/.bashrc`:
```bash
module load mpi/openmpi-x86_64
```

Then logout/login or run:
```bash
source ~/.bashrc
```

## Running on Department Server

### Basic test
```bash
make test
```

### Run with different process counts
```bash
mpirun -np 1 ./hw3    # 1 process
mpirun -np 2 ./hw3    # 2 processes
mpirun -np 4 ./hw3    # 4 processes
mpirun -np 8 ./hw3    # 8 processes
```

### Full benchmark suite
```bash
python3 performance_report.py
```

### Generate PDF report
```bash
python3 report.py
```

## Troubleshooting

### "command not found: module"
Your shell doesn't have module support. Try:
```bash
source /etc/profile.d/modules.sh
```

### "Python module not found"
Create a virtual environment:
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install fpdf2 matplotlib
```

### Still having issues?
Contact your instructor or system administrator about MPI availability.
