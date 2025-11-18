# Quick Start Guide - Running HW3 on the Cluster

## Option 1: Submit Batch Job (Recommended)
This runs everything automatically and emails you when done:

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

---

## Option 2: Interactive Session (For Testing)
If you're already in an interactive session:

```bash
module load openmpi/4.0.5-gcc10.2.0
./run_all.sh
```

Or start a new interactive session automatically:
```bash
interact -t 30:00 -N 1 --ntasks-per-node=8
# Then once inside:
module load openmpi/4.0.5-gcc10.2.0
./run_all.sh
```

---

## What Gets Generated

- `hw3.pdf` - Your complete report with 4 figures and analysis
- `hw3_output.log` - Full output log (batch mode only)
- `strong_scaling_results.csv` - Strong scaling data
- `weak_scaling_results.csv` - Weak scaling data  
- `openmp_results.csv` - OpenMP comparison data

---

## Multi-Node Testing (Optional)

For the multi-node figures, run with more nodes:

```bash
sbatch --nodes=2 --ntasks-per-node=16 RUN_ON_CLUSTER.sh
```

or

```bash
sbatch --nodes=4 --ntasks-per-node=16 RUN_ON_CLUSTER.sh
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
