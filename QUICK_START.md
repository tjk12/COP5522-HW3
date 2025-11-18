# Quick Start Guide - Running HW3 on the Cluster

## Step 1: Single-Node Tests (Required)
Run single-node benchmarks (1-8 cores):

```bash
sbatch RUN_ON_CLUSTER.sh
```

Check status:
```bash
squeue -u $USER
```

View results:
```bash
cat hw3_output.log
ls -lh hw3.pdf
```

**Result**: hw3.pdf with 2 figures populated (single-node) + 2 placeholders (multi-node)

---

## Step 2: Multi-Node Tests (Required for Complete Assignment)

**⚠️ IMPORTANT**: The assignment requires **4 figures** with actual data:
- Strong scaling: single-node AND multi-node
- Weak scaling: single-node AND multi-node

Run multi-node benchmarks (16-40 processes across 5 nodes):

```bash
sbatch RUN_MULTINODE.sh
```

This will:
1. Keep your existing single-node data (1-8 cores)
2. Add multi-node data (16, 24, 32, 40 cores)
3. Regenerate hw3.pdf with ALL 4 figures populated

---

## What Gets Generated

After **Step 1** (single-node only):
- `hw3.pdf` - Report with 2 real figures + 2 placeholders ⚠️
- `strong_scaling_results.csv` - Data for 1-8 cores
- `weak_scaling_results.csv` - Data for 1-8 cores
- `openmp_results.csv` - OpenMP comparison

After **Step 2** (multi-node added):
- `hw3.pdf` - Complete report with ALL 4 figures ✅
- `strong_scaling_results.csv` - Data for 1-8, 16, 24, 32, 40 cores
- `weak_scaling_results.csv` - Data for 1-8, 16, 24, 32, 40 cores

---

## Interactive Testing (Optional)

For quick testing:
```bash
interact -t 30:00 -N 1 --ntasks-per-node=8
module load openmpi/4.0.5-gcc10.2.0
./run_all.sh
```

---

## Troubleshooting

**If Python packages are missing:**
```bash
pip3 install --user 'numpy<1.20' 'pandas<1.2' 'matplotlib<3.4' 'fpdf2<2.6.0'
```

**If MPI not found:**
```bash
module load openmpi/4.0.5-gcc10.2.0
```

**Check what's available:**
```bash
module avail openmpi
```
