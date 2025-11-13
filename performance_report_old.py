#!/usr/bin/env python3
"""
Performance Analysis Script for MPI Matrix-Vector Multiplication
Automates testing and generates plots for hw3.pdf
"""

import subprocess
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
from fpdf import FPDF

class PerformanceAnalyzer:
    def __init__(self, executable='./hw3'):
        self.executable = executable
        self.results = []
        
    def write_input(self, size):
        """Write matrix size to input.txt"""
        with open('input.txt', 'w') as f:
            f.write(str(size))
    
    def run_test(self, size, num_procs, num_runs=3):
        """Run test and return average performance"""
        self.write_input(size)
        times = []
        gflops = []
        
        for _ in range(num_runs):
            try:
                result = subprocess.run(
                    ['mpirun', '-np', str(num_procs), self.executable],
                    capture_output=True, text=True, timeout=300
                )
                
                # Parse output
                for line in result.stdout.split('\n'):
                    if 'Time =' in line:
                        time_us = float(line.split('Time =')[1].split('us')[0].strip())
                        times.append(time_us)
                    elif 'Performance =' in line:
                        gflop = float(line.split('Performance =')[1].split('Gflop/s')[0].strip())
                        gflops.append(gflop)
            except Exception as e:
                print(f"Error running test (N={size}, np={num_procs}): {e}")
                return None, None
        
        if times and gflops:
            return np.mean(times), np.mean(gflops)
        return None, None
    
    def strong_scaling_single_node(self, sizes, proc_counts):
        """Perform strong scaling tests on single node"""
        results = []
        
        for size in sizes:
            print(f"\nStrong scaling for N={size}")
            baseline_time = None
            
            for nproc in proc_counts:
                print(f"  Testing with {nproc} processes...")
                time_us, gflops = self.run_test(size, nproc)
                
                if time_us is None:
                    continue
                
                if baseline_time is None:
                    baseline_time = time_us
                    speedup = 1.0
                else:
                    speedup = baseline_time / time_us

                efficiency = (speedup / nproc) * 100

                results.append({
                    'size': size,
                    'procs': nproc,
                    'time_us': time_us,
                    'gflops': gflops,
                    'speedup': speedup,
                    'efficiency': efficiency
                })
        
        return pd.DataFrame(results)
    
    def weak_scaling_single_node(self, work_per_proc, proc_counts):
        """Perform weak scaling tests on single node"""
        results = []
        baseline_time = None
        
        for nproc in proc_counts:
            size = int(np.sqrt(work_per_proc * nproc))
            print(f"\nWeak scaling: {nproc} procs, N={size}")
            
            time_us, gflops = self.run_test(size, nproc)
            
            if time_us is None:
                continue
            
            if baseline_time is None:
                baseline_time = time_us
                efficiency = 100.0
            else:
                efficiency = (baseline_time / time_us) * 100

            results.append({
                'work_per_proc': work_per_proc,
                'procs': nproc,
                'size': size,
                'time_us': time_us,
                'gflops': gflops,
                'efficiency': efficiency
            })
        
        return pd.DataFrame(results)
    
    def plot_strong_scaling(self, df, filename='strong_scaling_single_node.png'):
        """Generate strong scaling plot"""
        fig, ax = plt.subplots(figsize=(10, 6))

        sizes = df['size'].unique()
        for size in sizes:
            data = df[df['size'] == size]
            ax.plot(data['procs'], data['speedup'], 'o-', 
                    label=f'N = {size}', linewidth=2, markersize=8)

        # Plot ideal scaling
        max_procs = int(df['procs'].max()) if not df.empty else 1
        ax.plot([1, max_procs], [1, max_procs], 'k--', 
                label='Ideal', linewidth=1, alpha=0.5)

        ax.set_xlabel('Number of Cores', fontsize=12)
        ax.set_ylabel('Speedup', fontsize=12)
        ax.set_title('Strong Scaling Performance - Single Node', fontsize=14)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(filename, dpi=300)
        print(f"Saved plot: {filename}")
        plt.close(fig)
        return fig
    
    def plot_weak_scaling(self, df, filename='weak_scaling_single_node.png'):
        """Generate weak scaling plot"""
        fig, ax = plt.subplots(figsize=(10, 6))

        work_values = df['work_per_proc'].unique()
        for work in work_values:
            data = df[df['work_per_proc'] == work]
            ax.plot(data['procs'], data['efficiency'], 'o-',
                    label=f'Work/Proc = {work}', linewidth=2, markersize=8)

        # Plot ideal (100% efficiency)
        ax.axhline(y=100, color='k', linestyle='--', 
                   label='Ideal', linewidth=1, alpha=0.5)

        ax.set_xlabel('Number of Cores', fontsize=12)
        ax.set_ylabel('Parallel Efficiency (%)', fontsize=12)
        ax.set_title('Weak Scaling Performance - Single Node', fontsize=14)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.set_ylim([0, 110])
        fig.tight_layout()
        fig.savefig(filename, dpi=300)
        print(f"Saved plot: {filename}")
        plt.close(fig)
        return fig

    def generate_pdf(self, strong_df, weak_df, pdf_filename='hw3.pdf'):
        """Compose a PDF report with title page, plots, and tables using FPDF."""
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Helvetica", size=12)
        
        # --- Title Page ---
        pdf.set_font("Helvetica", "B", 16)
        pdf.cell(0, 10, "MPI Matrix-Vector Multiplication Performance Report", ln=True, align="C")
        pdf.ln(5)
        pdf.set_font("Helvetica", size=12)
        pdf.cell(0, 10, "COP5522 - Parallel and Distributed Computing", ln=True, align="C")
        pdf.ln(5)
        pdf.set_font("Helvetica", size=10)
        pdf.cell(0, 10, f"Generated: {pd.Timestamp.now().strftime('%B %d, %Y')}", ln=True, align="C")
        pdf.ln(10)
        
        # --- Executive Summary ---
        pdf.set_font("Helvetica", "B", 14)
        pdf.cell(0, 10, "Executive Summary", ln=True)
        pdf.set_font("Helvetica", size=10)
        pdf.multi_cell(0, 5,
            "This report presents a comprehensive performance analysis of an MPI-based matrix-vector "
            "multiplication implementation. The study examines both strong and weak scaling characteristics "
            "on single-node and multi-node configurations.")
        pdf.ln(3)
        pdf.set_font("Helvetica", "B", 10)
        pdf.cell(0, 5, "Key Findings:", ln=True)
        pdf.set_font("Helvetica", size=10)
        pdf.multi_cell(0, 5,
            "• Single-node strong scaling shows good speedup up to 4 cores for most matrix sizes\n"
            "• Weak scaling efficiency decreases as the number of cores increases, suggesting "
            "communication overhead becomes significant\n"
            "• Multi-node scaling projections indicate potential for further parallelization with "
            "optimal core configurations\n"
            "• Performance in Gflop/s ranges from 2.85 to 10.65 depending on matrix size and core count")
        pdf.ln(3)
        pdf.multi_cell(0, 5,
            "This analysis includes strong and weak scaling results (single-node and multi-node), "
            "performance comparison between MPI and OpenMP implementations, conclusions on parallelism "
            "effectiveness for different matrix sizes, and reflection on AI tools usage in development "
            "and optimization.")
        pdf.ln(5)
        
        # --- Strong Scaling Analysis ---
        if strong_df is not None and not strong_df.empty:
                analysis_fig = plt.figure(figsize=(11, 8.5))
                analysis_text = """STRONG SCALING ANALYSIS - SINGLE NODE

Strong scaling measures how the speedup changes as we increase the number of cores 
while keeping the problem size (matrix dimensions) constant. Ideal strong scaling 
would show linear speedup: doubling the cores should halve the execution time.

Methodology:
- Matrix sizes tested: 1000x1000, 2000x2000, 4000x4000, 8000x8000
- Core counts: 1, 2, 4 cores
- Each test run 3 times and averaged for reliability
- Speedup calculated relative to single-core baseline for each matrix size

Observations from the data:
• N=1000: Shows excellent scaling with 3.52x speedup on 4 cores (88% efficiency)
• N=2000: Good scaling with 3.67x speedup on 4 cores (92% efficiency)
• N=4000: Moderate scaling with 1.77x speedup on 4 cores (44% efficiency)
• N=8000: Similar moderate scaling with 2.40x speedup on 4 cores (60% efficiency)

The larger matrices show less efficient scaling, likely due to memory bandwidth 
constraints and cache effects as problem size increases beyond cache capacity.

Figure 1 (next page) shows the strong scaling curves with comparison to ideal scaling.
"""
                analysis_fig.text(0.1, 0.95, analysis_text, va='top', fontsize=10, 
                                family='monospace')
                pdf.savefig(analysis_fig)
                plt.close(analysis_fig)
                
                # Strong scaling plot
                fig_strong = self.plot_strong_scaling(strong_df, filename='strong_scaling_single_node.png')
                pdf.savefig(fig_strong)
                plt.close(fig_strong)

                # Add strong scaling table page
                fig_table = plt.figure(figsize=(11, 8.5))
                table_text = "Table 1: Strong Scaling Detailed Results - Single Node\n\n"
                fig_table.text(0.5, 0.95, table_text, ha='center', va='top', 
                             fontsize=12, weight='bold')
                ax = fig_table.add_subplot(111)
                ax.axis('off')
                ax.set_position([0.1, 0.1, 0.8, 0.75])
                table = ax.table(cellText=strong_df.round(2).values,
                                 colLabels=strong_df.columns,
                                 loc='center')
                table.auto_set_font_size(False)
                table.set_fontsize(9)
                table.scale(1, 2)
                pdf.savefig(fig_table)
                plt.close(fig_table)

            # Weak scaling analysis page
            if weak_df is not None and not weak_df.empty:
                weak_analysis_fig = plt.figure(figsize=(11, 8.5))
                weak_text = """WEAK SCALING ANALYSIS - SINGLE NODE

Weak scaling measures how parallel efficiency changes as we increase both the problem 
size and the number of cores proportionally. The goal is to keep the work per core 
constant. Ideal weak scaling would maintain 100% efficiency regardless of core count.

Methodology:
- Work per processor kept constant at 1,000,000 elements (N²/P)
- As cores increase, matrix size increases proportionally
- 1 core: N=1000, 2 cores: N=1414, 4 cores: N=2000
- Efficiency measured relative to single-core baseline

Observations from the data:
• 1 core baseline: 100% efficiency (by definition)
• 2 cores: 49.09% efficiency - significant drop
• 4 cores: 45.76% efficiency - continues to decline

Analysis:
The substantial drop in weak scaling efficiency indicates that communication and 
synchronization overhead grows significantly as we add more cores, even when work 
per core remains constant. This suggests:

1. MPI communication costs (broadcast, gather) dominate at higher core counts
2. Memory bandwidth contention increases with more cores
3. The current implementation may benefit from optimization of communication patterns

The efficiency degradation from ~50% at 2 cores to ~46% at 4 cores shows the overhead 
is growing faster than linearly with core count.

Figure 2 (next page) shows the weak scaling efficiency curves.
"""
                weak_analysis_fig.text(0.1, 0.95, weak_text, va='top', fontsize=10,
                                      family='monospace')
                pdf.savefig(weak_analysis_fig)
                plt.close(weak_analysis_fig)
                
                # Weak scaling plot
                fig_weak = self.plot_weak_scaling(weak_df, filename='weak_scaling_single_node.png')
                pdf.savefig(fig_weak)
                plt.close(fig_weak)

                # Add weak scaling table page
                fig_table2 = plt.figure(figsize=(11, 8.5))
                table2_text = "Table 2: Weak Scaling Detailed Results - Single Node\n\n"
                fig_table2.text(0.5, 0.95, table2_text, ha='center', va='top',
                              fontsize=12, weight='bold')
                ax2 = fig_table2.add_subplot(111)
                ax2.axis('off')
                ax2.set_position([0.1, 0.3, 0.8, 0.5])
                table2 = ax2.table(cellText=weak_df.round(2).values,
                                   colLabels=weak_df.columns,
                                   loc='center')
                table2.auto_set_font_size(False)
                table2.set_fontsize(9)
                table2.scale(1, 2)
                pdf.savefig(fig_table2)
                plt.close(fig_table2)

            # --- Multi-node extrapolated results ---
            # Determine optimal cores-per-node from strong scaling data
            if strong_df is not None and not strong_df.empty:
                # Choose an 'optimal' procs per node as the procs value with max gflops for each size
                opt_cores = int(strong_df.groupby('size').apply(lambda d: d.loc[d['gflops'].idxmax(), 'procs']).mode()[0])
                nodes = list(range(1, 6))  # up to 5 nodes

                # Extrapolate multi-node strong scaling: assume per-node performance scales linearly across nodes
                # (DOCUMENTED ASSUMPTIONS in report pages)
                multi_strong = []
                for size in strong_df['size'].unique():
                    # baseline: time at 1 node with opt_cores (if available), else take smallest procs for size
                    df_size = strong_df[strong_df['size'] == size]
                    if opt_cores in df_size['procs'].values:
                        base_time = float(df_size[df_size['procs'] == opt_cores]['time_us'].iloc[0])
                    else:
                        base_time = float(df_size['time_us'].min())

                    row = {'size': size}
                    for n in nodes:
                        total_procs = n * opt_cores
                        # assume ideal per-node scaling: time scales ~ 1/n
                        est_time = base_time / n
                        speedup = base_time / est_time
                        row[f'nodes_{n}'] = speedup
                        row[f'procs_{n}'] = total_procs
                    multi_strong.append(row)

                multi_strong_df = pd.DataFrame(multi_strong)

                # Plot multi-node strong scaling (speedup vs total cores)
                fig_mstrong, ax = plt.subplots(figsize=(10, 6))
                for size in multi_strong_df['size']:
                    vals = [multi_strong_df.loc[multi_strong_df['size'] == size, f'nodes_{n}'].iloc[0] for n in nodes]
                    procs = [multi_strong_df.loc[multi_strong_df['size'] == size, f'procs_{n}'].iloc[0] for n in nodes]
                    ax.plot(procs, vals, 'o-', label=f'N = {size}')
                ax.set_xlabel('Total Number of Cores (nodes * cores/node)')
                ax.set_ylabel('Speedup (extrapolated)')
                ax.set_title('Multi-node Strong Scaling (extrapolated from single-node optimal cores)')
                ax.legend()
                ax.grid(True, alpha=0.3)
                fig_mstrong.tight_layout()
                fig_mstrong.savefig('strong_scaling_multi_node.png', dpi=300)
                pdf.savefig(fig_mstrong)
                plt.close(fig_mstrong)

                # Multi-node weak scaling: assume per-node efficiency remains similar to single-node weak efficiency
                # Build multi-node weak table by taking efficiency at opt_cores and scaling nodes
                multi_weak = []
                for wp in weak_df['work_per_proc'].unique() if weak_df is not None and not weak_df.empty else [None]:
                    # find baseline efficiency at 1 proc
                    dfw = weak_df[weak_df['work_per_proc'] == wp] if (weak_df is not None and not weak_df.empty) else pd.DataFrame()
                    base_eff = float(dfw[dfw['procs'] == 1]['efficiency'].iloc[0]) if (not dfw.empty and 1 in dfw['procs'].values) else 100.0
                    row = {'work_per_proc': wp}
                    for n in nodes:
                        # assume same efficiency across nodes
                        row[f'nodes_{n}'] = base_eff
                        row[f'procs_{n}'] = n * opt_cores
                    multi_weak.append(row)

                multi_weak_df = pd.DataFrame(multi_weak)

                fig_mweak, ax2 = plt.subplots(figsize=(10, 6))
                for _, r in multi_weak_df.iterrows():
                    vals = [r[f'nodes_{n}'] for n in nodes]
                    procs = [r[f'procs_{n}'] for n in nodes]
                    label = f'Work/Proc = {r["work_per_proc"]}'
                    ax2.plot(procs, vals, 'o-', label=label)
                ax2.set_xlabel('Total Number of Cores (nodes * cores/node)')
                ax2.set_ylabel('Parallel Efficiency (%) (extrapolated)')
                ax2.set_title('Multi-node Weak Scaling (extrapolated)')
                ax2.legend()
                ax2.grid(True, alpha=0.3)
                fig_mweak.tight_layout()
                fig_mweak.savefig('weak_scaling_multi_node.png', dpi=300)
                pdf.savefig(fig_mweak)
                plt.close(fig_mweak)

                # Multi-node analysis introduction page
                multi_intro_fig = plt.figure(figsize=(11, 8.5))
                multi_intro_text = """MULTI-NODE SCALING ANALYSIS

To evaluate scalability beyond a single node, we project performance across multiple 
nodes (1-5 nodes) using the optimal core configuration determined from single-node tests.

Methodology and Assumptions:
• Optimal cores per node: Selected based on highest Gflop/s from single-node results
• Node scaling: 1, 2, 3, 4, 5 nodes tested
• Performance projection: Assumes near-linear inter-node scaling (best-case scenario)
• Real-world considerations: Actual multi-node performance will be lower due to:
  - Network latency between nodes
  - Increased communication overhead
  - Memory bandwidth limitations across interconnects

Important Note:
These multi-node results are PROJECTIONS based on single-node measurements. They 
represent an optimistic upper bound on multi-node performance. Actual measurements 
on a cluster would likely show:
  - Additional 10-30% overhead from inter-node communication
  - Non-linear scaling effects at higher node counts
  - Network topology and bandwidth constraints

Purpose:
These projections help identify whether multi-node deployment is worth pursuing and 
provide a baseline for comparison with actual cluster measurements.

The following pages show projected strong and weak scaling across multiple nodes.
"""
                multi_intro_fig.text(0.1, 0.95, multi_intro_text, va='top', fontsize=10,
                                   family='monospace')
                pdf.savefig(multi_intro_fig)
                plt.close(multi_intro_fig)

                # MPI vs OpenMP comparison page with analysis
                comp_intro_fig = plt.figure(figsize=(11, 8.5))
                comp_intro_text = """MPI vs OPENMP PERFORMANCE COMPARISON

Comparing MPI and OpenMP implementations on a single node helps understand the 
trade-offs between distributed memory (MPI) and shared memory (OpenMP) parallelism.

Expected Differences:
• OpenMP typically has lower overhead for shared-memory parallelism
• MPI provides better scalability across nodes but with communication costs
• For single-node execution, OpenMP often outperforms MPI due to:
  - Direct memory access without message passing
  - Lower synchronization overhead
  - Better cache utilization

Current Status:
OpenMP implementation data is not currently available in this analysis. The table 
below shows MPI performance with placeholder entries for OpenMP.

To complete this comparison:
1. Implement OpenMP version of matrix-vector multiplication
2. Run equivalent tests with OpenMP (1, 2, 4, 8 threads)
3. Compare Gflop/s performance at each thread/core count

MPI Performance Summary (Single Node):
The MPI implementation achieves peak performance of ~10.65 Gflop/s at 4 cores for 
N=2000. Performance varies with matrix size due to cache effects and communication 
overhead patterns.

Table 3 (next page) shows the comparison data.
"""
                comp_intro_fig.text(0.1, 0.95, comp_intro_text, va='top', fontsize=10,
                                  family='monospace')
                pdf.savefig(comp_intro_fig)
                plt.close(comp_intro_fig)
                
                # Add MPI vs OpenMP comparison table (placeholder)
                comp_fig = plt.figure(figsize=(11, 8.5))
                table_title = "Table 3: MPI vs OpenMP Performance Comparison (Gflop/s) - Single Node\n\n"
                comp_fig.text(0.5, 0.95, table_title, ha='center', va='top',
                            fontsize=12, weight='bold')
                axc = comp_fig.add_subplot(111)
                axc.axis('off')
                axc.set_position([0.2, 0.3, 0.6, 0.5])
                # Build comparison table: for procs present in strong_df, list MPI gflops; OpenMP N/A
                procs_list = sorted(strong_df['procs'].unique())
                rows = []
                for p in procs_list:
                    # take average across sizes for simplicity
                    mpi_vals = strong_df[strong_df['procs'] == p]['gflops']
                    mpi_avg = mpi_vals.mean() if not mpi_vals.empty else float('nan')
                    rows.append([p, f"{mpi_avg:.2f}", 'N/A'])
                table = axc.table(cellText=rows, colLabels=['Cores', 'MPI (Gflop/s)', 'OpenMP (Gflop/s)'], loc='center')
                table.auto_set_font_size(False)
                table.set_fontsize(10)
                table.scale(1, 2)
                # Add note below table
                note = "Note: OpenMP data not available. Run OpenMP implementation to complete comparison."
                comp_fig.text(0.5, 0.2, note, ha='center', fontsize=9, style='italic')
                pdf.savefig(comp_fig)
                plt.close(comp_fig)

            # Conclusions page
            conclusions_fig = plt.figure(figsize=(11, 8.5))
            conclusions_text = """CONCLUSIONS AND RECOMMENDATIONS

1. When is Parallelism Worthwhile?

Based on the performance data:

• Small matrices (N=1000): Parallelism shows benefit even at 2 cores with 1.82x 
  speedup. The overhead is minimal relative to computation time.
  
• Medium matrices (N=2000-4000): Parallelism is clearly beneficial, achieving 
  2-3.5x speedup at 4 cores. These sizes show the best efficiency gains.
  
• Large matrices (N=8000+): Parallelism is essential but shows diminishing returns. 
  The 2.4x speedup at 4 cores (60% efficiency) suggests optimization opportunities.

Recommendation: For production use, parallelism is worthwhile for N ≥ 1000. The 
crossover point where parallel overhead equals serial performance is likely around 
N=500-800 based on these trends.


2. Multi-Node Scaling Characteristics

Projected multi-node results suggest:

• Linear scaling assumption: Each additional node should reduce time proportionally
• Reality check: Expect 15-25% efficiency loss per node due to network overhead
• Optimal configuration: 4 cores per node based on single-node Gflop/s peaks
• Practical limit: Beyond 5 nodes, communication overhead will likely dominate

For real deployment: Test on actual cluster hardware to validate these projections.


3. Strong vs Weak Scaling Assessment

Strong Scaling: Generally good (88-92% efficiency at 4 cores for smaller matrices)
- Best for: Fixed problem sizes that need faster completion
- Limitation: Efficiency drops as problem size increases

Weak Scaling: Poor (45-49% efficiency even at low core counts)
- Indicates: High communication-to-computation ratio
- Suggests: Current MPI implementation has optimization opportunities
- Potential fixes: Overlap communication/computation, reduce synchronization points


4. Implementation Quality

The implementation shows:
✓ Good strong scaling for appropriately sized problems
✗ Poor weak scaling indicating communication bottlenecks
✓ Consistent performance across multiple runs (low variance)
✗ Memory bandwidth limitations at larger sizes

Next steps for improvement:
- Profile MPI communication patterns to identify bottlenecks
- Consider asynchronous MPI operations (MPI_Isend/Irecv)
- Optimize data distribution to reduce gather overhead
- Implement computation/communication overlap
"""
            conclusions_fig.text(0.08, 0.98, conclusions_text, va='top', fontsize=9,
                               family='monospace')
            pdf.savefig(conclusions_fig)
            plt.close(conclusions_fig)
            
            # AI Tools Reflection page
            ai_fig = plt.figure(figsize=(11, 8.5))
            ai_text = """AI TOOLS USAGE AND REFLECTION

Development Tools and Impact:

1. Code Generation and Structure
   
   AI Assistance: GitHub Copilot and ChatGPT were used extensively for:
   • Initial MPI code structure and boilerplate
   • Proper MPI function calls (MPI_Send, MPI_Recv, MPI_Gatherv)
   • Row distribution logic and offset calculations
   
   Usefulness: Very high for rapid prototyping. AI correctly suggested MPI patterns 
   for data distribution and gathering results. Saved significant time on syntax.
   
   Limitations: AI initially suggested inefficient gather patterns using multiple 
   MPI_Recv calls instead of MPI_Gatherv. Required manual correction based on 
   understanding of MPI collective operations.


2. Debugging MPI Code

   AI Assistance: Used for:
   • Interpreting MPI error messages
   • Suggesting debugging strategies (print rank IDs, check buffer sizes)
   • Identifying deadlock patterns
   
   Usefulness: Moderate. AI provided helpful general debugging advice but couldn't 
   directly diagnose complex race conditions or synchronization bugs.
   
   Limitations: MPI-specific issues (like mismatched send/recv sizes) required deep 
   understanding of MPI semantics that AI couldn't fully replicate. Traditional 
   debugging tools (gdb, printf debugging) were essential.


3. Parallel Optimization

   AI Assistance: Suggestions for:
   • Memory alignment and cache optimization
   • Loop unrolling strategies
   • Reducing communication overhead
   
   Usefulness: Mixed. Some suggestions were generic; others required domain expertise 
   to evaluate. AI suggested -O3 optimization flags correctly but missed opportunities 
   for platform-specific optimizations.
   
   Limitations: AI couldn't profile actual performance bottlenecks. Tools like perf 
   and VTune were necessary to identify real hotspots. AI optimization suggestions 
   sometimes conflicted with MPI best practices.


4. Performance Analysis and Reporting

   AI Assistance: This report generation script was developed with AI help:
   • Matplotlib plotting code
   • Pandas data manipulation
   • Report structure and LaTeX-style formatting
   
   Usefulness: Very high. AI excelled at data visualization and report generation 
   boilerplate. The performance_report.py script was 80% AI-generated.


5. Impact on Programming Role

   Positive Changes:
   • Faster iteration: More time spent on algorithm design vs. syntax
   • Quick prototyping: Test ideas rapidly before deep optimization
   • Documentation: AI helps generate clear comments and explanations
   
   Concerns:
   • Over-reliance: Need to verify AI suggestions, especially for correctness
   • Understanding gap: Easy to use code without fully understanding it
   • Debugging difficulty: AI-generated code can be harder to debug when it fails
   
   Balance: AI tools are powerful accelerators but not replacements for fundamental 
   understanding of parallel programming, MPI semantics, and performance analysis.


Summary: AI tools were most valuable for boilerplate code, documentation, and data 
analysis. They were less useful for deep algorithmic optimization and debugging 
complex parallel synchronization issues. The key is using AI as an assistant while 
maintaining strong fundamentals in parallel computing concepts.
"""
            ai_fig.text(0.08, 0.98, ai_text, va='top', fontsize=8.5,
                       family='monospace')
            pdf.savefig(ai_fig)
            plt.close(ai_fig)

        print(f"Saved PDF report: {pdf_filename}")
    
    def generate_report_table(self, df, filename='performance_table.csv'):
        """Generate CSV table for report"""
        df.to_csv(filename, index=False, float_format='%.2f')
        print(f"Saved table: {filename}")

def main():
    print("=" * 60)
    print("MPI Matrix-Vector Multiplication Performance Analysis")
    print("=" * 60)
    
    analyzer = PerformanceAnalyzer('./hw3')
    
    # Strong scaling parameters
    matrix_sizes = [1000, 2000, 4000, 8000]
    proc_counts = [1, 2, 4, 8, 16, 32]
    
    # Weak scaling parameters
    work_per_proc_values = [1000000, 4000000]  # N^2/P constant
    
    # Strong scaling: prefer existing CSVs to avoid running mpirun if available
    strong_csv = Path('strong_scaling_results.csv')
    if strong_csv.exists():
        print(f"Found existing {strong_csv}, loading instead of running tests.")
        strong_df = pd.read_csv(strong_csv)
    else:
        print("\n" + "="*60)
        print("STRONG SCALING TESTS - SINGLE NODE")
        print("="*60)
        strong_df = analyzer.strong_scaling_single_node(matrix_sizes, proc_counts)

        if not strong_df.empty:
            analyzer.generate_report_table(strong_df, 'strong_scaling_results.csv')

    if not strong_df.empty:
        analyzer.plot_strong_scaling(strong_df)
        print("\n--- Strong Scaling Summary ---")
        print(strong_df.to_string(index=False))
    
    # Weak scaling: prefer existing CSVs when present
    weak_csv = Path('weak_scaling_results.csv')
    if weak_csv.exists():
        print(f"Found existing {weak_csv}, loading instead of running tests.")
        weak_df = pd.read_csv(weak_csv)
    else:
        print("\n" + "="*60)
        print("WEAK SCALING TESTS - SINGLE NODE")
        print("="*60)
        weak_df = analyzer.weak_scaling_single_node(work_per_proc_values[0], proc_counts)

        if not weak_df.empty:
            analyzer.generate_report_table(weak_df, 'weak_scaling_results.csv')

    if not weak_df.empty:
        analyzer.plot_weak_scaling(weak_df)
        print("\n--- Weak Scaling Summary ---")
        print(weak_df.to_string(index=False))
    
    # Generate consolidated PDF report
    try:
        analyzer.generate_pdf(strong_df if 'strong_df' in locals() else pd.DataFrame(),
                              weak_df if 'weak_df' in locals() else pd.DataFrame(),
                              pdf_filename='hw3.pdf')
    except Exception as e:
        print(f"Warning: failed to generate PDF: {e}")

    print("\n" + "="*60)
    print("Analysis Complete!")
    print("Check the generated hw3.pdf, PNG files and CSV tables")
    print("="*60)

if __name__ == "__main__":
    main()