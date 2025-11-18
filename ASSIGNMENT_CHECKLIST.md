# Assignment Requirements Checklist

## ✅ Code Requirements

- [x] Process 0 reads matrix size from `input.txt`
- [x] Uses send/receive (not broadcast) to distribute N
- [x] Uses user-defined MPI datatype for sending N
- [x] Each process computes part of matrix-vector product
- [x] Uses collective communication (MPI_Gatherv) to collect results
- [x] Process 0 prints result and timing
- [x] Efficient memory usage (each process stores only its rows)
- [x] Timing excludes file I/O and initial sends/receives

## ⚠️ Report Requirements (hw3.pdf)

### Figures (4 required)
- [x] **Figure 1**: Strong scaling - single node (1-8 cores) ✅ HAS DATA
- [x] **Figure 2**: Strong scaling - multi-node (16-40 cores) ⚠️ NEEDS RUN
- [x] **Figure 3**: Weak scaling - single node (1-8 cores) ✅ HAS DATA
- [x] **Figure 4**: Weak scaling - multi-node (16-40 cores) ⚠️ NEEDS RUN
- [x] All figures have captions and labels

### Tables & Analysis
- [x] Table comparing MPI vs OpenMP performance (Gflop/s) ✅
- [x] Conclusions about when parallelism is worthwhile ✅
- [x] Analysis of speedup on multiple nodes ⚠️ NEEDS MULTI-NODE DATA
- [x] Weak scaling analysis ✅
- [x] Strong scaling analysis ✅

### AI Tool Reflection
- [x] Document AI tool usage for coding/debugging ✅
- [x] Reflect on usefulness for MPI code ✅
- [x] Discuss limitations ✅

## 📋 What You Need to Do

**Current Status**: Single-node tests complete (Figures 1 & 3 populated)

**To Complete Assignment**:
1. Run multi-node tests to populate Figures 2 & 4:
   ```bash
   sbatch RUN_MULTINODE.sh
   ```

2. This will:
   - Test with 16, 24, 32, 40 processes across 5 nodes
   - Keep your existing single-node data
   - Regenerate hw3.pdf with ALL 4 figures populated
   - Add multi-node analysis to conclusions

3. Verify final hw3.pdf has:
   - ✅ 4 figures with real data (no placeholders)
   - ✅ Multi-node speedup analysis
   - ✅ Comparison of single-node vs multi-node efficiency

## 📊 Expected Results

### Strong Scaling
- **Y-axis**: Speedup (T₁/Tₚ)
- **X-axis**: Number of cores
- **Curves**: One per matrix size (N=1000, 2000, 4000, 8000)
- **Single-node**: 1, 2, 4, 8 cores
- **Multi-node**: 16, 24, 32, 40 cores

### Weak Scaling  
- **Y-axis**: Parallel efficiency (Gflop/s ratio)
- **X-axis**: Number of cores
- **Curves**: Different work per process values
- **Single-node**: 1, 2, 4, 8 cores
- **Multi-node**: 16, 24, 32, 40 cores

## 🎯 Grading Criteria

1. **Correctness** (mandatory)
   - [x] Code uses specified MPI features
   - [x] Correct matrix-vector multiplication

2. **Performance** 
   - [x] No unusually inefficient implementations
   - [x] Good sequential baseline (10+ Gflop/s) ✅

3. **Presentation**
   - [x] Clear figures with proper labels
   - ⚠️ ALL 4 figures with actual data (not placeholders)
   - [x] Insightful conclusions based on results

## ⏱️ Time Estimate

- Multi-node benchmark run: ~30-60 minutes
- Total wall time to completion: 1 hour

## 🚀 Final Command

```bash
sbatch RUN_MULTINODE.sh
# Wait for completion
cat hw3_multinode.log
# Verify hw3.pdf has all 4 figures populated
```
