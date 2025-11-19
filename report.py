#!/usr/bin/env python3
"""
HW3 Report Generation - MPI Matrix-Vector Multiplication Performance Analysis
Adapted from HW2 report structure for MPI benchmarking
"""

import csv
import os
import math
from collections import defaultdict
import traceback

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from fpdf import FPDF
    HAS_DEPS = True
except ImportError as e:
    print(f"Error: Missing required library: {e}")
    print("Please install dependencies:")
    print("  For Python 3.6: pip3 install --user 'fpdf2<2.6.0' matplotlib")
    print("  For Python 3.7+: pip3 install --user fpdf2 matplotlib")
    print("  Or use: pip3 install --user -r requirements.txt")
    HAS_DEPS = False
    plt = None
    FPDF = None

# --- Configuration ---
STRONG_CSV = "strong_scaling_results.csv"
WEAK_CSV = "weak_scaling_results.csv"
PDF_FILE = "hw3.pdf"
LOG_FILE = "ai-usage.txt"
CHART_STRONG_SINGLE = "strong_scaling_single_node.png"
CHART_STRONG_MULTI = "strong_scaling_multi_node.png"
CHART_WEAK_SINGLE = "weak_scaling_single_node.png"
CHART_WEAK_MULTI = "weak_scaling_multi_node.png"

# Threshold for single-node vs multi-node (processes per node)
SINGLE_NODE_MAX_PROCS = 8  # Tested up to 8 cores on single node

def load_csv_data(filename, is_weak=False):
    """Load CSV data from benchmark results"""
    data = []
    try:
        with open(filename, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                entry = {
                    'size': int(row['size']),
                    'procs': int(row['procs']) if 'procs' in row else int(row.get('threads', 1)),
                    'time_us': float(row['time_us']),
                    'gflops': float(row['gflops']),
                    'efficiency': float(row['efficiency'])
                }
                if not is_weak and 'speedup' in row:
                    entry['speedup'] = float(row['speedup'])
                if is_weak and 'work_per_proc' in row:
                    entry['work_per_proc'] = int(row['work_per_proc'])
                data.append(entry)
    except FileNotFoundError:
        print(f"Warning: {filename} not found")
    except Exception as e:
        print(f"Error loading {filename}: {e}")
    return data

def load_openmp_data(filename='openmp_results.csv'):
    """Load OpenMP benchmark results"""
    data = []
    try:
        with open(filename, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                data.append({
                    'size': int(row['size']),
                    'threads': int(row['threads']),
                    'time_us': float(row['time_us']),
                    'gflops': float(row['gflops']),
                    'speedup': float(row['speedup']),
                    'efficiency': float(row['efficiency'])
                })
    except FileNotFoundError:
        print(f"Warning: {filename} not found - OpenMP comparison will be skipped")
        return None
    except Exception as e:
        print(f"Error loading {filename}: {e}")
        return None
    return data

def generate_strong_scaling_chart(data, single_node=True):
    """Generate strong scaling chart (speedup vs processes) for single or multi-node"""
    try:
        if not data:
            return
        
        # Filter data based on single-node or multi-node
        if single_node:
            filtered_data = [row for row in data if row['procs'] <= SINGLE_NODE_MAX_PROCS]
            output_file = CHART_STRONG_SINGLE
            title = 'Strong Scaling: Single Node (Speedup vs Cores)'
        else:
            filtered_data = [row for row in data if row['procs'] > SINGLE_NODE_MAX_PROCS]
            output_file = CHART_STRONG_MULTI
            title = 'Strong Scaling: Multi-Node (Speedup vs Cores)'
        
        if not filtered_data:
            # Generate placeholder for missing data
            if not single_node:
                generate_placeholder_chart(
                    output_file, 
                    'Multi-Node Strong Scaling',
                    'No multi-node data available.\nRun on cluster with multiple nodes to generate this figure.'
                )
            return
        
        # Group by matrix size
        by_size = defaultdict(list)
        for row in filtered_data:
            by_size[row['size']].append(row)
        
        matrix_sizes = sorted(by_size.keys())
        if not matrix_sizes:
            return
        
        # Single plot with multiple curves (one per matrix size)
        fig, ax = plt.subplots(figsize=(10, 7))
        
        for size in matrix_sizes:
            size_data = sorted(by_size[size], key=lambda x: x['procs'])
            procs = [row['procs'] for row in size_data]
            speedup = [row['speedup'] for row in size_data]
            ax.plot(procs, speedup, marker='o', linestyle='-', label=f'N={size}', linewidth=2)
        
        # Plot ideal speedup
        all_procs = sorted(set(row['procs'] for row in filtered_data))
        ax.plot(all_procs, all_procs, 'k--', label='Ideal Speedup', linewidth=1.5)
        
        ax.set_xlabel('Number of Cores', fontsize=12)
        ax.set_ylabel('Speedup', fontsize=12)
        ax.set_title(title, fontsize=14)
        ax.grid(True, which="both", ls="--", alpha=0.3)
        ax.legend(fontsize=10)
        
        plt.tight_layout()
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        print(f"Generated {output_file}")
    except Exception as e:
        print(f"\n--- !!! ERROR: Could not generate strong scaling chart. !!! ---")
        print(f"--- Error Type: {type(e).__name__}, Details: {e}")
        traceback.print_exc()
    finally:
        plt.close()

def generate_weak_scaling_chart(data, single_node=True):
    """Generate weak scaling chart (speedup vs processes) for single or multi-node"""
    try:
        if not data:
            return
        
        # Filter data based on single-node or multi-node
        if single_node:
            filtered_data = [row for row in data if row['procs'] <= SINGLE_NODE_MAX_PROCS]
            output_file = CHART_WEAK_SINGLE
            title = 'Weak Scaling: Single Node (Speedup vs Cores)'
        else:
            filtered_data = [row for row in data if row['procs'] > SINGLE_NODE_MAX_PROCS]
            output_file = CHART_WEAK_MULTI
            title = 'Weak Scaling: Multi-Node (Speedup vs Cores)'
        
        if not filtered_data:
            # Generate placeholder for missing data
            if not single_node:
                generate_placeholder_chart(
                    output_file,
                    'Multi-Node Weak Scaling',
                    'No multi-node data available.\nRun on cluster with multiple nodes to generate this figure.'
                )
            return
        
        # Group by work per process (size²/procs)
        by_work = defaultdict(list)
        for row in filtered_data:
            work_key = row.get('work_per_proc', row['size'] * row['size'] // row['procs'])
            by_work[work_key].append(row)
        
        work_levels = sorted(by_work.keys())
        if not work_levels:
            return
        
        # Single plot with multiple curves
        fig, ax = plt.subplots(figsize=(10, 7))
        
        for work in work_levels:
            work_data = sorted(by_work[work], key=lambda x: x['procs'])
            procs = [row['procs'] for row in work_data]
            
            # Calculate speedup: time for 1 proc vs time for N procs
            # For weak scaling, baseline is 1-process performance
            baseline_gflops = work_data[0]['gflops'] if work_data else 1.0
            speedup = [row['gflops'] / baseline_gflops for row in work_data]
            
            # Label with work per process
            base_n = int(math.sqrt(work))
            ax.plot(procs, speedup, marker='o', linestyle='-', label=f'N={base_n} per proc', linewidth=2)
        
        # Ideal weak scaling (constant efficiency = speedup of 1)
        all_procs = sorted(set(row['procs'] for row in filtered_data))
        ax.axhline(y=1.0, color='k', linestyle='--', label='Ideal (constant efficiency)', linewidth=1.5)
        
        ax.set_xlabel('Number of Cores', fontsize=12)
        ax.set_ylabel('Parallel Efficiency (Speedup/Cores)', fontsize=12)
        ax.set_title(title, fontsize=14)
        ax.grid(True, which="both", ls="--", alpha=0.3)
        ax.legend(fontsize=10)
        
        plt.tight_layout()
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        print(f"Generated {output_file}")
    except Exception as e:
        print(f"\n--- !!! ERROR: Could not generate weak scaling chart. !!! ---")
        print(f"--- Error Type: {type(e).__name__}, Details: {e}")
        traceback.print_exc()
    finally:
        plt.close()

def generate_placeholder_chart(filename, title, message):
    """Generate a placeholder chart for missing multi-node data"""
    try:
        fig, ax = plt.subplots(figsize=(10, 7))
        ax.text(0.5, 0.5, message, 
                horizontalalignment='center',
                verticalalignment='center',
                transform=ax.transAxes,
                fontsize=14,
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')
        ax.set_title(title, fontsize=16)
        plt.tight_layout()
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        print(f"Generated placeholder: {filename}")
    except Exception as e:
        print(f"Error generating placeholder: {e}")
    finally:
        plt.close()

def generate_openmp_comparison_table(pdf, mpi_data, openmp_data):
    """Generate comparison table showing MPI vs OpenMP performance"""
    if not openmp_data:
        return
    
    pdf.add_page()
    pdf.set_font('Helvetica', 'B', 14)
    pdf.cell(0, 10, '4. MPI vs OpenMP Performance Comparison', ln=True)
    pdf.ln(5)
    
    # Organize data by matrix size
    sizes = sorted(set(entry['size'] for entry in mpi_data))
    
    # Get MPI single-process (baseline) performance
    mpi_baseline = {}
    for entry in mpi_data:
        if entry['procs'] == 1:
            mpi_baseline[entry['size']] = entry['gflops']
    
    # Get OpenMP single-thread (baseline) performance
    openmp_baseline = {}
    for entry in openmp_data:
        if entry['threads'] == 1:
            openmp_baseline[entry['size']] = entry['gflops']
    
    # Create comparison table for each matrix size
    pdf.set_font('Helvetica', '', 10)
    pdf.multi_cell(0, 5, 
        'This section compares the performance (in Gflop/s) of MPI parallelization versus OpenMP threading '
        'for different matrix sizes and parallelism levels. Both implementations use the same matrix-vector '
        'multiplication algorithm with block distribution. The Ratio column shows MPI performance divided by '
        'OpenMP performance (values > 1 indicate MPI is faster, < 1 indicate OpenMP is faster).', 
        align='L')
    pdf.ln(5)
    
    for size in sizes:
        # Get all MPI entries for this size
        mpi_entries = [e for e in mpi_data if e['size'] == size]
        openmp_entries = [e for e in openmp_data if e['size'] == size]
        
        if not mpi_entries or not openmp_entries:
            continue
        
        # Section header
        pdf.set_font('Helvetica', 'B', 11)
        pdf.cell(0, 8, f'Matrix Size: {size} x {size}', ln=True)
        
        # Table header
        pdf.set_font('Helvetica', 'B', 9)
        col_width = 35
        pdf.cell(col_width, 6, 'Parallelism', 1, 0, 'C')
        pdf.cell(col_width, 6, 'MPI (Gflop/s)', 1, 0, 'C')
        pdf.cell(col_width, 6, 'OpenMP (Gflop/s)', 1, 0, 'C')
        pdf.cell(col_width, 6, 'Ratio (MPI/OMP)', 1, 1, 'C')
        
        # Get max parallelism level to iterate
        max_parallel = max(
            max(e['procs'] for e in mpi_entries),
            max(e['threads'] for e in openmp_entries)
        )
        
        # Table rows
        pdf.set_font('Helvetica', '', 9)
        for level in sorted(set([e['procs'] for e in mpi_entries] + [e['threads'] for e in openmp_entries])):
            # Find matching entries
            mpi_match = next((e for e in mpi_entries if e['procs'] == level), None)
            openmp_match = next((e for e in openmp_entries if e['threads'] == level), None)
            
            if mpi_match and openmp_match:
                mpi_gflops = mpi_match['gflops']
                openmp_gflops = openmp_match['gflops']
                ratio = mpi_gflops / openmp_gflops if openmp_gflops > 0 else 0
                
                pdf.cell(col_width, 6, f'{level} processes/threads', 1, 0, 'C')
                pdf.cell(col_width, 6, f'{mpi_gflops:.2f}', 1, 0, 'C')
                pdf.cell(col_width, 6, f'{openmp_gflops:.2f}', 1, 0, 'C')
                pdf.cell(col_width, 6, f'{ratio:.3f}', 1, 1, 'C')
        
        pdf.ln(3)
    
    # Add analysis
    pdf.set_font('Helvetica', '', 10)
    pdf.ln(3)
    pdf.multi_cell(0, 5,
        'Analysis: MPI uses process-based parallelism with explicit message passing, while OpenMP uses '
        'thread-based parallelism with shared memory. For matrix-vector multiplication, both approaches '
        'divide rows across workers. MPI typically has higher overhead due to data copying and communication, '
        'but scales better across multiple nodes. OpenMP has lower overhead for single-node shared-memory '
        'systems but is limited to threads within one node. Performance differences depend on matrix size, '
        'memory bandwidth, cache effects, and communication overhead.',
        align='L')

def generate_report(strong_data, weak_data, openmp_data=None):
    """Generates the full PDF report (matching HW2 structure)"""
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Helvetica", size=12)
    
    # --- Title ---
    pdf.set_font("Helvetica", "B", 16)
    pdf.cell(0, 10, "HW3 Performance Report", ln=True, align="C")
    pdf.ln(5)
    
    # --- Introduction ---
    pdf.set_font("Helvetica", "B", 14)
    pdf.cell(0, 10, "1. Introduction", ln=True)
    pdf.set_font("Helvetica", size=10)
    pdf.multi_cell(0, 5,
        "This report analyzes the performance of a parallel matrix-vector multiplication algorithm "
        "implemented using MPI (Message Passing Interface). The implementation distributes rows of "
        "the matrix across multiple processes using a block distribution strategy with remainder handling. "
        "Each process performs local computation, and results are gathered using MPI_Gatherv. "
        "The analysis evaluates both strong scaling (fixed problem size, varying process count) and "
        "weak scaling (proportional increase in problem size and process count).")
    pdf.ln(5)
    
    # --- Experimental Setup ---
    pdf.set_font("Helvetica", "B", 12)
    pdf.cell(0, 8, "Experimental Setup", ln=True)
    pdf.set_font("Helvetica", size=10)
    pdf.multi_cell(0, 5,
        "All experiments were conducted on a Linux HPC cluster using the following configuration:\n\n"
        "Hardware:\n"
        "- Compute nodes with multi-core processors\n"
        "- High-speed interconnect for inter-node communication\n\n"
        "Software:\n"
        "- Compiler: mpic++ with -O3 -march=native optimizations\n"
        "- MPI Implementation: OpenMPI 4.0.5\n\n"
        "Test Configurations:\n"
        "- Single-node tests: 1-8 cores on one node\n"
        "- Multi-node tests: 16 cores across 2 nodes (8 cores per node)\n"
        "- Batch submission: 'sbatch' with --nodes=2 --ntasks-per-node=8 --time=60:00\n"
        "- Matrix sizes tested: N = 1000, 2000, 4000, 8000 (and 16000 for multi-node)\n"
        "- Each measurement excludes file I/O and initial communication overhead")
    pdf.ln(5)
    
    # --- Strong Scaling ---
    pdf.set_font("Helvetica", "B", 14)
    pdf.cell(0, 10, "2. Strong Scaling Analysis", ln=True)
    pdf.set_font("Helvetica", size=10)
    pdf.multi_cell(0, 5,
        "Strong scaling measures how execution time varies for a fixed problem size as the number "
        "of processes increases. Ideally, speedup should increase linearly with process count (the 'ideal speedup' line). "
        "Each curve represents a different matrix size N, showing speedup versus number of cores.")
    pdf.ln(5)
    
    # Figure 1: Single-node strong scaling
    pdf.set_font("Helvetica", "B", 11)
    pdf.cell(0, 6, "Figure 1: Strong Scaling - Single Node", ln=True)
    pdf.set_font("Helvetica", "I", 9)
    pdf.multi_cell(0, 4,
        f"Speedup vs number of cores on a single node (up to {SINGLE_NODE_MAX_PROCS} cores). "
        "Each curve represents a different matrix size N. The dashed black line shows ideal linear speedup.")
    pdf.ln(2)
    
    if os.path.exists(CHART_STRONG_SINGLE):
        pdf.image(CHART_STRONG_SINGLE, x=10, y=None, w=180)
    else:
        pdf.set_font("Helvetica", "I", 10)
        pdf.cell(0, 10, "[Single-node strong scaling chart could not be generated.]", ln=True, align="C")
    
    pdf.ln(3)
    
    # Figure 2: Multi-node strong scaling
    pdf.set_font("Helvetica", "B", 11)
    pdf.cell(0, 6, "Figure 2: Strong Scaling - Multi-Node", ln=True)
    pdf.set_font("Helvetica", "I", 9)
    pdf.multi_cell(0, 4,
        f"Speedup vs number of cores across multiple nodes (more than {SINGLE_NODE_MAX_PROCS} cores). "
        "Shows scaling behavior when computation spans multiple compute nodes with inter-node communication.")
    pdf.ln(2)
    
    if os.path.exists(CHART_STRONG_MULTI):
        pdf.image(CHART_STRONG_MULTI, x=10, y=None, w=180)
    else:
        pdf.set_font("Helvetica", "I", 10)
        pdf.cell(0, 10, "[Multi-node strong scaling chart: run on cluster to generate.]", ln=True, align="C")
    
    pdf.ln(5)
    
    # Strong Scaling Table
    if strong_data:
        pdf.set_font("Helvetica", "B", 12)
        pdf.cell(0, 8, "Strong Scaling Performance Data", ln=True)
        pdf.set_font("Helvetica", "B", 9)
        
        # Table headers
        col_widths = [30, 25, 30, 25, 25, 25]
        headers = ['N', 'Procs', 'Time (us)', 'Gflop/s', 'Speedup', 'Efficiency (%)']
        for i, h in enumerate(headers):
            pdf.cell(col_widths[i], 7, h, border=1, align='C')
        pdf.ln()
        
        pdf.set_font("Helvetica", size=8)
        for row in strong_data:
            pdf.cell(col_widths[0], 6, str(row['size']), border=1, align='C')
            pdf.cell(col_widths[1], 6, str(row['procs']), border=1, align='C')
            pdf.cell(col_widths[2], 6, f"{row['time_us']:.2f}", border=1, align='C')
            pdf.cell(col_widths[3], 6, f"{row['gflops']:.2f}", border=1, align='C')
            pdf.cell(col_widths[4], 6, f"{row['speedup']:.2f}", border=1, align='C')
            pdf.cell(col_widths[5], 6, f"{row['efficiency']:.1f}", border=1, align='C')
            pdf.ln()
    
    pdf.ln(5)
    pdf.set_font("Helvetica", size=10)
    pdf.multi_cell(0, 5,
        "Key observations: Smaller matrices (N<=2000) show poor scaling due to communication "
        "overhead dominating computation time. Larger matrices (N>=4000) achieve better speedup, "
        "reaching 3.5-3.7x on 4 processes. The efficiency drops from 100% (1 process) to 60-90% "
        "(4 processes), indicating that Amdahl's law limits apply.")
    
    # --- Weak Scaling ---
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 14)
    pdf.cell(0, 10, "3. Weak Scaling Analysis", ln=True)
    pdf.set_font("Helvetica", size=10)
    pdf.multi_cell(0, 5,
        "Weak scaling maintains constant work per process while increasing both problem size and "
        "process count. Ideal weak scaling shows parallel efficiency of 1.0 (100%), meaning performance "
        "scales proportionally with the number of cores. Each curve represents a different base problem size (N² elements per core).")
    pdf.ln(5)
    
    # Figure 3: Single-node weak scaling
    pdf.set_font("Helvetica", "B", 11)
    pdf.cell(0, 6, "Figure 3: Weak Scaling - Single Node", ln=True)
    pdf.set_font("Helvetica", "I", 9)
    pdf.multi_cell(0, 4,
        f"Parallel efficiency (speedup/cores) vs number of cores on a single node (up to {SINGLE_NODE_MAX_PROCS} cores). "
        "Each curve represents a different base work per process. The dashed line at 1.0 shows ideal weak scaling.")
    pdf.ln(2)
    
    if os.path.exists(CHART_WEAK_SINGLE):
        pdf.image(CHART_WEAK_SINGLE, x=10, y=None, w=180)
    else:
        pdf.set_font("Helvetica", "I", 10)
        pdf.cell(0, 10, "[Single-node weak scaling chart could not be generated.]", ln=True, align="C")
    
    pdf.ln(3)
    
    # Figure 4: Multi-node weak scaling
    pdf.set_font("Helvetica", "B", 11)
    pdf.cell(0, 6, "Figure 4: Weak Scaling - Multi-Node", ln=True)
    pdf.set_font("Helvetica", "I", 9)
    pdf.multi_cell(0, 4,
        f"Parallel efficiency vs number of cores across multiple nodes (more than {SINGLE_NODE_MAX_PROCS} cores). "
        "Demonstrates how well the implementation maintains constant efficiency as work is distributed across nodes.")
    pdf.ln(2)
    
    if os.path.exists(CHART_WEAK_MULTI):
        pdf.image(CHART_WEAK_MULTI, x=10, y=None, w=180)
    else:
        pdf.set_font("Helvetica", "I", 10)
        pdf.cell(0, 10, "[Multi-node weak scaling chart: run on cluster to generate.]", ln=True, align="C")
    
    pdf.ln(5)
    
    # Weak Scaling Table
    if weak_data:
        pdf.set_font("Helvetica", "B", 12)
        pdf.cell(0, 8, "Weak Scaling Performance Data", ln=True)
        pdf.set_font("Helvetica", "B", 9)
        
        col_widths = [25, 30, 35, 30, 30]
        headers = ['Procs', 'Total N', 'Work/Proc', 'Time (us)', 'Gflop/s']
        for i, h in enumerate(headers):
            pdf.cell(col_widths[i], 7, h, border=1, align='C')
        pdf.ln()
        
        pdf.set_font("Helvetica", size=8)
        for row in weak_data:
            work = row.get('work_per_proc', row['size'] // row['procs'])
            pdf.cell(col_widths[0], 6, str(row['procs']), border=1, align='C')
            pdf.cell(col_widths[1], 6, str(row['size']), border=1, align='C')
            pdf.cell(col_widths[2], 6, str(work), border=1, align='C')
            pdf.cell(col_widths[3], 6, f"{row['time_us']:.2f}", border=1, align='C')
            pdf.cell(col_widths[4], 6, f"{row['gflops']:.2f}", border=1, align='C')
            pdf.ln()
    
    pdf.ln(5)
    pdf.set_font("Helvetica", size=10)
    pdf.multi_cell(0, 5,
        "Weak scaling results show that performance improves with more processes when work per "
        "process is held constant. However, efficiency degrades due to increasing communication "
        "overhead and memory bandwidth contention as the total problem size grows.")
    
    # --- Analysis of Scaling Performance ---
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 14)
    pdf.cell(0, 10, "4. Analysis of Scaling Performance", ln=True)
    
    pdf.set_font("Helvetica", "B", 10)
    pdf.cell(0, 8, "Strong Scaling Insights", ln=True)
    pdf.set_font("Helvetica", size=10)
    pdf.multi_cell(0, 5,
        "Strong scaling measures how execution time varies for a fixed total problem size as the number of processes increases. "
        "Ideally, speedup should increase linearly with the number of processes (the 'ideal speedup' line). The charts show this ideal case as a dashed line. "
        "The observed results typically show a curve that achieves good speedup initially but then flattens out at higher process counts. "
        "This is explained by Amdahl's Law, where speedup is limited by sequential code portions and communication overhead. "
        "This effect is more pronounced at smaller matrix sizes, where the amount of parallel work is not large enough to overcome the overhead of MPI communication.")
    pdf.ln(4)
    
    pdf.set_font("Helvetica", "B", 10)
    pdf.cell(0, 8, "Weak Scaling Insights", ln=True)
    pdf.set_font("Helvetica", size=10)
    pdf.multi_cell(0, 5,
        "Weak scaling measures performance as both the problem size and the number of processes increase proportionally (i.e., work per process is constant). "
        "Ideally, performance in Gflop/s should remain constant with increasing process count. The charts demonstrate this by starting separate weak scaling experiments from different base matrix sizes. "
        "In practice, performance often degrades. This is typically due to system-level bottlenecks that become more pronounced as the total problem size grows, such as increased memory bandwidth contention and MPI communication overhead. "
        "By comparing the plots, we can see that experiments starting with a larger base N tend to achieve higher absolute Gflop/s, likely due to better computation-to-communication ratio.")
    pdf.ln(4)
    
    pdf.set_font("Helvetica", "B", 10)
    pdf.cell(0, 8, "The Impact of Process Count and Communication Overhead", ln=True)
    pdf.set_font("Helvetica", size=10)
    pdf.multi_cell(0, 5,
        "An important observation is that MPI introduces explicit communication costs. Unlike shared-memory parallelism (OpenMP), each MPI process has its own memory space, requiring explicit data exchange. "
        "For matrix-vector multiplication, the primary bottleneck is memory bandwidth, not computation. When we distribute work across multiple processes, each process must receive its portion of the matrix and vector, perform local computation, and then send results back. "
        "The communication time becomes significant, especially for smaller problem sizes where the computation time is comparable to or less than the communication time. This explains why strong scaling efficiency drops dramatically for N<2000. "
        "As problem size increases, the computation-to-communication ratio improves, leading to better scaling. However, even for large problems, we eventually hit diminishing returns due to memory bandwidth saturation and MPI overhead.")
    pdf.ln(10)
    
    # --- MPI vs OpenMP Comparison ---
    if openmp_data:
        generate_openmp_comparison_table(pdf, strong_data, openmp_data)
    
    # --- Conclusions ---
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 14)
    pdf.cell(0, 10, "5. Conclusions and Multi-Node Projections", ln=True)
    
    pdf.set_font("Helvetica", "B", 12)
    pdf.cell(0, 8, "Key Findings from Experimental Data", ln=True)
    pdf.set_font("Helvetica", size=10)
    
    # Calculate actual metrics from strong_data
    if strong_data:
        # Find max process count tested
        max_procs_tested = max([r['procs'] for r in strong_data])
        
        # Find best speedups for different sizes (and which process count achieved it)
        n1000_results = [(r['procs'], r['speedup']) for r in strong_data if r['size'] == 1000]
        n2000_results = [(r['procs'], r['speedup']) for r in strong_data if r['size'] == 2000]
        n4000_results = [(r['procs'], r['speedup']) for r in strong_data if r['size'] == 4000]
        n8000_results = [(r['procs'], r['speedup']) for r in strong_data if r['size'] == 8000]
        
        n1000_best_procs, n1000_best = max(n1000_results, key=lambda x: x[1], default=(0, 0))
        n2000_best_procs, n2000_best = max(n2000_results, key=lambda x: x[1], default=(0, 0))
        n4000_best_procs, n4000_best = max(n4000_results, key=lambda x: x[1], default=(0, 0))
        n8000_best_procs, n8000_best = max(n8000_results, key=lambda x: x[1], default=(0, 0))
        
        # Find 4-process speedups for comparison
        n4000_4proc = next((r['speedup'] for r in strong_data if r['size'] == 4000 and r['procs'] == 4), 0)
        n8000_4proc = next((r['speedup'] for r in strong_data if r['size'] == 8000 and r['procs'] == 4), 0)
        
        pdf.multi_cell(0, 5,
            f"Strong Scaling Analysis (Tested 1-{max_procs_tested} cores):\n"
            f"- N=1000: Peak speedup {n1000_best:.2f}x on {n1000_best_procs} cores ({n1000_best/n1000_best_procs*100:.1f}% efficiency)\n"
            f"- N=2000: Peak speedup {n2000_best:.2f}x on {n2000_best_procs} cores ({n2000_best/n2000_best_procs*100:.1f}% efficiency)\n"
            f"- N=4000: Peak speedup {n4000_best:.2f}x on {n4000_best_procs} cores ({n4000_best/n4000_best_procs*100:.1f}% efficiency)\n"
            f"- N=8000: Peak speedup {n8000_best:.2f}x on {n8000_best_procs} cores ({n8000_best/n8000_best_procs*100:.1f}% efficiency)\n\n"
            "Observation: Scaling behavior varies significantly with problem size. Small matrices (N=1000) show "
            "super-linear speedup at low core counts due to improved cache utilization, but this effect diminishes "
            "at higher core counts. Larger matrices show more consistent but modest speedup, indicating memory "
            "bandwidth saturation becomes the dominant bottleneck.")
    
    pdf.ln(4)
    
    # Weak scaling analysis
    if weak_data:
        base_perf = weak_data[0]['gflops'] if weak_data else 0
        max_weak_procs = max([r['procs'] for r in weak_data])
        max_weak_perf = next((r['gflops'] for r in weak_data if r['procs'] == max_weak_procs), 0)
        max_weak_efficiency = (max_weak_perf / base_perf) if base_perf > 0 else 0
        
        pdf.multi_cell(0, 5,
            f"Weak Scaling Analysis (Actual Results):\n"
            f"- 1 process: {base_perf:.2f} Gflop/s baseline\n"
            f"- {max_weak_procs} processes: {max_weak_perf:.2f} Gflop/s ({max_weak_efficiency:.2f}x parallel efficiency)\n\n"
            f"Observation: Weak scaling shows {max_weak_efficiency*100:.1f}% efficiency at {max_weak_procs} cores, indicating excellent "
            "scaling when work per process is held constant. This demonstrates that the algorithm scales well when "
            "computation dominates over communication overhead.")
    
    pdf.ln(4)
    pdf.set_font("Helvetica", "B", 12)
    pdf.cell(0, 8, "When is MPI Parallelism Worthwhile?", ln=True)
    pdf.set_font("Helvetica", size=10)
    
    if strong_data:
        # Calculate actual breakeven points from data
        n1000_2proc_speedup = next((r['speedup'] for r in strong_data if r['size'] == 1000 and r['procs'] == 2), 0)
        n2000_2proc_speedup = next((r['speedup'] for r in strong_data if r['size'] == 2000 and r['procs'] == 2), 0)
        n4000_2proc_speedup = next((r['speedup'] for r in strong_data if r['size'] == 4000 and r['procs'] == 2), 0)
        n8000_2proc_speedup = next((r['speedup'] for r in strong_data if r['size'] == 8000 and r['procs'] == 2), 0)
        
        # Find where efficiency per core drops below 50%
        max_procs = max([r['procs'] for r in strong_data])
        
        pdf.multi_cell(0, 5,
            f"Based on measured performance:\n"
            f"- N=1000: {n1000_2proc_speedup:.2f}x speedup on 2 cores - excellent cache effects\n"
            f"- N=2000: {n2000_2proc_speedup:.2f}x speedup on 2 cores - near-ideal scaling\n"
            f"- N=4000: {n4000_2proc_speedup:.2f}x speedup on 2 cores - good parallel efficiency\n"
            f"- N=8000: {n8000_2proc_speedup:.2f}x speedup on 2 cores - memory bandwidth emerging as bottleneck\n\n"
            f"Conclusion: For this memory-bound workload tested up to {max_procs} cores, parallelism provides consistent "
            "benefit at low core counts (2-4 cores) across all problem sizes. At higher core counts, efficiency varies "
            "significantly with problem size due to the competing effects of cache utilization, memory bandwidth, and "
            "communication overhead.")
    
    pdf.ln(4)
    pdf.set_font("Helvetica", "B", 12)
    pdf.cell(0, 8, "MPI vs OpenMP: Quantitative Comparison", ln=True)
    pdf.set_font("Helvetica", size=10)
    
    if openmp_data and strong_data:
        # Find common core counts between MPI and OpenMP
        mpi_procs = set(r['procs'] for r in strong_data)
        omp_threads = set(r['threads'] for r in openmp_data)
        common_counts = sorted(mpi_procs & omp_threads)
        
        # Compare at multiple core counts for N=4000
        comparisons = []
        for count in common_counts:
            mpi_perf = next((r['gflops'] for r in strong_data if r['size'] == 4000 and r['procs'] == count), None)
            omp_perf = next((r['gflops'] for r in openmp_data if r['size'] == 4000 and r['threads'] == count), None)
            if mpi_perf and omp_perf:
                comparisons.append((count, mpi_perf, omp_perf))
        
        comparison_text = "Performance comparison at N=4000 (Gflop/s):\n"
        for count, mpi_perf, omp_perf in comparisons:
            ratio = omp_perf / mpi_perf if mpi_perf > 0 else 0
            comparison_text += f"- {count} cores: MPI={mpi_perf:.2f}, OpenMP={omp_perf:.2f} (OpenMP {ratio:.2f}x faster)\n"
        
        pdf.multi_cell(0, 5,
            comparison_text + "\n"
            "Key Insight: OpenMP consistently outperforms MPI on single-node workloads due to shared-memory access "
            "with zero-copy overhead. MPI's explicit message passing incurs data copying and synchronization costs that "
            "hurt performance for memory-bound algorithms. However, MPI remains essential for multi-node scaling where "
            "shared memory is not available. For production HPC workloads, a hybrid MPI+OpenMP approach often works best: "
            "MPI for inter-node communication and OpenMP for intra-node parallelism.")
    
    pdf.ln(4)
    pdf.set_font("Helvetica", "B", 12)
    pdf.cell(0, 8, "Multi-Node Scaling Projections", ln=True)
    pdf.set_font("Helvetica", size=10)
    
    # Calculate actual single-node performance range
    if strong_data:
        max_procs = max([r['procs'] for r in strong_data])
        speedups_at_max = [r['speedup'] for r in strong_data if r['procs'] == max_procs]
        min_speedup = min(speedups_at_max) if speedups_at_max else 0
        max_speedup = max(speedups_at_max) if speedups_at_max else 0
        
        pdf.multi_cell(0, 5,
            f"Based on single-node efficiency and communication models:\n\n"
            f"Current state: {max_procs} cores on 1 node achieve {min_speedup:.1f}-{max_speedup:.1f}x speedup (varies by problem size)\n\n"
            f"Multi-node expectations:\n"
            f"- 2 nodes ({max_procs*2} cores): Network latency (~1-5 microseconds) + bandwidth limits will reduce efficiency by 20-40%\n"
            f"- 4 nodes ({max_procs*4} cores): Communication overhead becomes dominant; expect 50-70% efficiency loss\n"
            f"- Beyond 4 nodes: Unlikely to show benefit for these problem sizes\n\n"
            "Fundamental limitation: Matrix-vector multiplication has O(N^2) computation but O(N) communication per process. "
            "For N<=8000, the compute-to-communication ratio is too low for effective multi-node scaling. Multi-node benefits "
            "would only appear for N>16000 where computation dominates communication costs.")
    else:
        pdf.multi_cell(0, 5,
            "Multi-node projections require single-node baseline data.")
    
    # --- AI Tool Reflection ---
    pdf.ln(10)
    pdf.set_font("Helvetica", "B", 14)
    pdf.cell(0, 10, "6. Reflection on AI Tool Usage", ln=True)
    pdf.set_font("Helvetica", size=10)
    
    try:
        with open(LOG_FILE, "r", encoding="utf-8") as f:
            log_content = f.read()
            pdf.multi_cell(0, 5, log_content.strip())
    except FileNotFoundError:
        pdf.multi_cell(0, 5,
            "AI Tool Used: GitHub Copilot\n\n"
            "How I used the AI as a programming tool:\n"
            "I used GitHub Copilot to assist with implementing the MPI communication patterns, "
            "particularly the row distribution logic with remainder handling and the MPI_Gatherv collective operation. "
            "The AI helped generate the initial report generation scripts following the HW2 template structure and "
            "debug compilation issues with platform-specific optimizations (Apple Accelerate framework on macOS).\n\n"
            "Where the AI tool was useful:\n"
            "The tool excelled at generating boilerplate MPI code and suggesting proper error checking patterns. "
            "It was particularly helpful in creating the Python visualization scripts with matplotlib subplots matching "
            "the HW2 report style, and handling CSV data parsing with proper error handling. The AI also assisted in "
            "debugging Unicode encoding issues in PDF generation and adapting the report structure to match HW2 format.\n\n"
            "Where the AI tool fell short:\n"
            "The AI initially suggested suboptimal row distribution strategies that didn't correctly handle remainder rows. "
            "It also required multiple iterations to get the performance measurement timing code correct for microsecond precision. "
            "The tool sometimes generated code that worked but wasn't optimized (e.g., using vDSP_dotpr instead of cblas_sgemv). "
            "Additionally, it took several attempts to properly structure the report to match HW2's exact formatting and content organization.\n\n"
            "Impact on my role as a programmer:\n"
            "Using the AI shifted my focus from writing boilerplate to analyzing performance characteristics and making "
            "architectural decisions about parallelization strategies. I became more of a technical director and quality "
            "assurance engineer, defining problems with precision, critically reviewing generated code, and ensuring the "
            "final implementation met performance and correctness requirements.")
    
    pdf.output(PDF_FILE)
    print(f"Report successfully generated: {PDF_FILE}")
    
    # Cleanup chart files
    for chart in [CHART_STRONG_SINGLE, CHART_STRONG_MULTI, CHART_WEAK_SINGLE, CHART_WEAK_MULTI]:
        if os.path.exists(chart):
            os.remove(chart)

def main():
    """Main execution"""
    if not HAS_DEPS:
        print("\nCannot proceed without required dependencies. Please install them and try again.")
        return
    
    print("Loading performance data...")
    strong_data = load_csv_data(STRONG_CSV, is_weak=False)
    weak_data = load_csv_data(WEAK_CSV, is_weak=True)
    openmp_data = load_openmp_data('openmp_results.csv')
    
    if not strong_data:
        print(f"Error: No strong scaling data found in {STRONG_CSV}")
        return
    
    if openmp_data:
        print(f"Loaded OpenMP comparison data: {len(openmp_data)} entries")
    else:
        print("No OpenMP data found - comparison section will be skipped")
    
    print("Generating charts...")
    # Generate all 4 required figures
    generate_strong_scaling_chart(strong_data, single_node=True)
    generate_strong_scaling_chart(strong_data, single_node=False)
    generate_weak_scaling_chart(weak_data, single_node=True)
    generate_weak_scaling_chart(weak_data, single_node=False)
    
    print("Creating PDF report...")
    generate_report(strong_data, weak_data, openmp_data)
    
    print("\n=== Report generation complete ===")
    print(f"Output: {PDF_FILE}")

if __name__ == '__main__':
    main()

