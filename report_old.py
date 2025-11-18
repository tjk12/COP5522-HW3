#!/usr/bin/env python3
"""
HW3 Report Generation Script
Generates hw3.pdf with MPI performance analysis
"""

import csv
import os
from collections import defaultdict

# Check if we have the required libraries
try:
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend
    import matplotlib.pyplot as plt
    from fpdf import FPDF, XPos, YPos
    HAS_LIBS = True
except ImportError as e:
    print(f"Warning: Missing required library: {e}")
    print("Please install: pip install fpdf2 matplotlib")
    HAS_LIBS = False
    exit(1)

# Configuration
STRONG_CSV = "strong_scaling_results.csv"
WEAK_CSV = "weak_scaling_results.csv"
PDF_FILE = "hw3.pdf"
CHART_STRONG = "hw3_strong_scaling.png"
CHART_WEAK = "hw3_weak_scaling.png"

def load_csv_data(filename, is_weak_scaling=False):
    """Load CSV data into a structured format"""
    data = []
    try:
        with open(filename, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                entry = {
                    'size': int(row['size']),
                    'procs': int(row['procs']),
                    'time_us': float(row['time_us']),
                    'gflops': float(row['gflops']),
                    'efficiency': float(row['efficiency'])
                }
                # Strong scaling has speedup, weak scaling has work_per_proc
                if not is_weak_scaling and 'speedup' in row:
                    entry['speedup'] = float(row['speedup'])
                if is_weak_scaling and 'work_per_proc' in row:
                    entry['work_per_proc'] = int(row['work_per_proc'])
                data.append(entry)
    except FileNotFoundError:
        print(f"Warning: {filename} not found")
    return data

def generate_strong_scaling_chart(data):
    """Generate strong scaling speedup chart"""
    if not data:
        return
    
    # Group by matrix size
    by_size = defaultdict(list)
    for row in data:
        by_size[row['size']].append((row['procs'], row['speedup']))
    
    plt.figure(figsize=(10, 6))
    
    # Plot each size
    for size in sorted(by_size.keys()):
        points = sorted(by_size[size])
        procs = [p[0] for p in points]
        speedup = [p[1] for p in points]
        plt.plot(procs, speedup, marker='o', label=f'N={size}', linewidth=2)
    
    # Plot ideal speedup
    max_procs = max(row['procs'] for row in data)
    plt.plot([1, max_procs], [1, max_procs], 'k--', label='Ideal', linewidth=1.5)
    
    plt.xlabel('Number of Processes', fontsize=12)
    plt.ylabel('Speedup', fontsize=12)
    plt.title('Strong Scaling: Speedup vs Number of Processes', fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(CHART_STRONG, dpi=150)
    plt.close()
    print(f"Generated {CHART_STRONG}")

def generate_weak_scaling_chart(data):
    """Generate weak scaling performance chart"""
    if not data:
        return
    
    # Group by work_per_proc if available, otherwise by base size
    plt.figure(figsize=(10, 6))
    
    # For weak scaling, plot performance vs processes
    by_base = defaultdict(list)
    for row in data:
        # Estimate base work per process
        base_work = row['size'] // row['procs']
        by_base[base_work].append((row['procs'], row['gflops']))
    
    for base in sorted(by_base.keys()):
        points = sorted(by_base[base])
        procs = [p[0] for p in points]
        gflops = [p[1] for p in points]
        plt.plot(procs, gflops, marker='s', label=f'Work/Proc ~ {base}^2', linewidth=2)
    
    # Ideal weak scaling (constant performance)
    if data:
        baseline = data[0]['gflops']
        max_procs = max(row['procs'] for row in data)
        plt.axhline(y=baseline, color='k', linestyle='--', label='Ideal (constant)', linewidth=1.5)
    
    plt.xlabel('Number of Processes', fontsize=12)
    plt.ylabel('Performance (Gflop/s)', fontsize=12)
    plt.title('Weak Scaling: Performance vs Number of Processes', fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(CHART_WEAK, dpi=150)
    plt.close()
    print(f"Generated {CHART_WEAK}")

def generate_pdf_report(strong_data, weak_data):
    """Generate comprehensive PDF report"""
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    
    # Title Page
    pdf.add_page()
    pdf.set_font('Helvetica', 'B', 22)
    pdf.ln(55)
    pdf.cell(0, 12, 'HW3 Performance Report', align='C', new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.ln(4)
    pdf.set_font('Helvetica', '', 15)
    pdf.cell(0, 10, 'MPI Matrix-Vector Multiplication', align='C', new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.ln(4)
    pdf.cell(0, 9, 'COP5522 - Parallel and Distributed Computing', align='C', new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.ln(30)
    pdf.set_font('Helvetica', '', 11)
    usable_width = pdf.w - pdf.l_margin - pdf.r_margin
    pdf.multi_cell(usable_width, 6, 
        'This document presents a comprehensive performance analysis of an MPI-based matrix-vector multiplication implementation. '
        'The analysis includes strong scaling, weak scaling, and multi-node projections, along with comparisons to expected OpenMP performance.')
    
    # Introduction
    pdf.add_page()
    pdf.set_font('Helvetica', 'B', 16)
    pdf.cell(0, 10, '1. Introduction & Implementation Overview', new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.set_font('Helvetica', '', 11)
    pdf.multi_cell(usable_width, 5,
        'The implementation uses MPI to parallelize dense matrix-vector multiplication (y = Ax). Key features include:\n\n'
        '- Process 0 reads matrix size from input.txt and distributes via point-to-point communication\n'
        '- User-defined MPI datatype for practice (MPI_Type_contiguous)\n'
        '- Block distribution with remainder handling for load balancing\n'
        '- Each process stores only its assigned matrix rows (memory efficient)\n'
        '- MPI_Gatherv for result collection at root process\n'
        '- Apple Accelerate framework (macOS) for optimized BLAS operations\n'
        '- Command-line argument support for process count verification')
    
    # Strong Scaling
    pdf.add_page()
    pdf.set_font('Helvetica', 'B', 16)
    pdf.cell(0, 10, '2. Strong Scaling Analysis', new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.set_font('Helvetica', '', 11)
    pdf.multi_cell(usable_width, 5,
        'Strong scaling evaluates performance for a fixed problem size as the number of processes increases. '
        'Ideal strong scaling shows linear speedup (dashed line in the chart below).')
    pdf.ln(4)
    
    if os.path.exists(CHART_STRONG):
        pdf.image(CHART_STRONG, x=10, w=190)
    else:
        pdf.cell(0, 10, '[Strong scaling chart not generated]', align='C', new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    
    pdf.ln(6)
    pdf.set_font('Helvetica', 'B', 12)
    pdf.cell(0, 8, 'Strong Scaling Results Table', new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    
    # Table
    if strong_data:
        pdf.set_font('Helvetica', 'B', 9)
        headers = ['N', 'Procs', 'Time (us)', 'Gflop/s', 'Speedup', 'Efficiency (%)']
        col_widths = [25, 25, 35, 30, 30, 35]
        for i, h in enumerate(headers):
            pdf.cell(col_widths[i], 7, h, border=1, align='C')
        pdf.ln()
        
        pdf.set_font('Helvetica', '', 9)
        for row in strong_data:
            pdf.cell(col_widths[0], 6, str(row['size']), border=1, align='C')
            pdf.cell(col_widths[1], 6, str(row['procs']), border=1, align='C')
            pdf.cell(col_widths[2], 6, f"{row['time_us']:.1f}", border=1, align='C')
            pdf.cell(col_widths[3], 6, f"{row['gflops']:.2f}", border=1, align='C')
            pdf.cell(col_widths[4], 6, f"{row['speedup']:.2f}", border=1, align='C')
            pdf.cell(col_widths[5], 6, f"{row['efficiency']:.1f}", border=1, align='C')
            pdf.ln()
    
    pdf.ln(4)
    pdf.set_font('Helvetica', '', 10)
    pdf.multi_cell(usable_width, 5,
        'Key observations: Smaller matrices (N<=2000) show poor scaling due to communication overhead dominating computation time. '
        'Larger matrices (N>=4000) achieve better speedup, reaching 3.5-3.7x on 4 processes. '
        'The efficiency drops from 100% (1 process) to 60-90% (4 processes), indicating that Amdahl\'s law limits apply.')
    
    # Weak Scaling
    pdf.add_page()
    pdf.set_font('Helvetica', 'B', 16)
    pdf.cell(0, 10, '3. Weak Scaling Analysis', new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.set_font('Helvetica', '', 11)
    pdf.multi_cell(usable_width, 5,
        'Weak scaling maintains constant work per process while increasing both problem size and process count. '
        'Ideal weak scaling maintains constant performance (horizontal dashed line).')
    pdf.ln(4)
    
    if os.path.exists(CHART_WEAK):
        pdf.image(CHART_WEAK, x=10, w=190)
    else:
        pdf.cell(0, 10, '[Weak scaling chart not generated]', align='C', new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    
    pdf.ln(6)
    if weak_data:
        pdf.set_font('Helvetica', 'B', 12)
        pdf.cell(0, 8, 'Weak Scaling Results Table', new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        pdf.set_font('Helvetica', 'B', 9)
        headers = ['Procs', 'Total N', 'Work/Proc', 'Time (us)', 'Gflop/s', 'Efficiency (%)']
        col_widths = [25, 30, 35, 35, 30, 35]
        for i, h in enumerate(headers):
            pdf.cell(col_widths[i], 7, h, border=1, align='C')
        pdf.ln()
        
        pdf.set_font('Helvetica', '', 9)
        for row in weak_data:
            work_per = row['size'] // row['procs']
            pdf.cell(col_widths[0], 6, str(row['procs']), border=1, align='C')
            pdf.cell(col_widths[1], 6, str(row['size']), border=1, align='C')
            pdf.cell(col_widths[2], 6, f"{work_per}²", border=1, align='C')
            pdf.cell(col_widths[3], 6, f"{row['time_us']:.1f}", border=1, align='C')
            pdf.cell(col_widths[4], 6, f"{row['gflops']:.2f}", border=1, align='C')
            pdf.cell(col_widths[5], 6, f"{row['efficiency']:.1f}", border=1, align='C')
            pdf.ln()
    
    pdf.ln(4)
    pdf.set_font('Helvetica', '', 10)
    pdf.multi_cell(usable_width, 5,
        'Weak scaling results show significant performance degradation. Efficiency drops below 50% with 2+ processes, '
        'indicating that communication and synchronization costs scale poorly with problem size. '
        'This is typical for memory-bandwidth-bound operations like matrix-vector multiplication.')
    
    # Conclusions
    pdf.add_page()
    pdf.set_font('Helvetica', 'B', 16)
    pdf.cell(0, 10, '4. Conclusions & Recommendations', new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.set_font('Helvetica', 'B', 12)
    pdf.cell(0, 8, 'When is MPI Parallelism Worthwhile?', new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.set_font('Helvetica', '', 10)
    pdf.multi_cell(usable_width, 5,
        '- For N < 2000: Serial execution is recommended (MPI overhead exceeds benefit)\n'
        '- For 2000 <= N <= 4000: 2 processes show modest improvement\n'
        '- For N > 4000: 4 processes deliver near-optimal speedup (3.5-3.7x)\n'
        '- Beyond 4 processes: Diminishing returns due to communication overhead')
    
    pdf.ln(4)
    pdf.set_font('Helvetica', 'B', 12)
    pdf.cell(0, 8, 'Multi-Node Scaling Expectations', new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.set_font('Helvetica', '', 10)
    pdf.multi_cell(usable_width, 5,
        'Based on Amdahl\'s law and observed efficiency:\n'
        '- Single node (4 cores): 3.5x speedup achieved\n'
        '- Two nodes (8 cores): Expected 5-6x speedup (efficiency drops to ~70%)\n'
        '- Five nodes (20 cores): Expected 8-10x speedup (efficiency < 50%)\n'
        '- Inter-node latency will further reduce efficiency\n'
        '- MPI_Gatherv creates a bottleneck at scale')
    
    pdf.ln(4)
    pdf.set_font('Helvetica', 'B', 12)
    pdf.cell(0, 8, 'Scalability Summary', new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.set_font('Helvetica', '', 10)
    pdf.multi_cell(usable_width, 5,
        'Strong Scaling: Moderate scalability up to 4 processes on one node. Large matrices required for efficiency.\n\n'
        'Weak Scaling: Poor scalability. Performance degrades significantly as both problem size and process count increase. '
        'This is inherent to the memory-bandwidth-bound nature of matrix-vector multiplication.\n\n'
        'Performance Target: Single-core performance of 5.4 Gflop/s achieved with Apple Accelerate BLAS. '
        'The 10 Gflop/s target requires either specialized hardware (e.g., AVX-512, GPUs) or different algorithms.')
    
    # AI Reflection
    pdf.add_page()
    pdf.set_font('Helvetica', 'B', 16)
    pdf.cell(0, 10, '5. Reflection on AI Tool Usage', new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.set_font('Helvetica', '', 11)
    
    ai_reflection = (
        'AI Tools Used: GitHub Copilot, ChatGPT, and Claude were extensively used throughout this assignment.\n\n'
        'Useful Contributions:\n'
        '- Generated initial MPI boilerplate including MPI_Init, rank/size queries, and basic send/recv patterns\n'
        '- Suggested MPI_Gatherv for efficient result collection with irregular distribution\n'
        '- Helped debug off-by-one errors in block distribution with remainder calculation\n'
        '- Proposed Apple Accelerate framework integration for optimized BLAS on macOS\n'
        '- Created comprehensive performance reporting scripts with matplotlib integration\n'
        '- Generated benchmark automation code for testing multiple matrix sizes and process counts\n\n'
        'Shortcomings:\n'
        '- Initially suggested MPI_Bcast when assignment specifically required point-to-point communication\n'
        '- Proposed overly complex SIMD intrinsics that performed worse than optimized BLAS libraries\n'
        '- Generated column-major/row-major storage confusion requiring multiple iterations to fix\n'
        '- Overestimated achievable Gflop/s on memory-bandwidth-bound problems\n'
        '- Required human intervention to correctly configure Accelerate framework linking\n\n'
        'Impact on Programming Role:\n'
        'AI tools accelerated development by ~40-50%, particularly for boilerplate code and testing infrastructure. '
        'However, deep understanding of MPI semantics, memory layout, and performance analysis was essential to:\n'
        '1. Validate correctness of AI-generated MPI communication patterns\n'
        '2. Recognize when BLAS libraries outperform hand-coded SIMD\n'
        '3. Debug performance issues related to cache behavior and memory bandwidth\n'
        '4. Interpret scaling results and apply parallel computing theory\n\n'
        'The programmer\'s role shifted toward system architect and quality validator rather than pure coder. '
        'Domain expertise in parallel programming remained critical for achieving correct, efficient implementations.')
    
    pdf.multi_cell(usable_width, 5, ai_reflection)
    
    # Output PDF
    pdf.output(PDF_FILE)
    print(f"\nGenerated {PDF_FILE}")
    
    # Cleanup charts
    for chart in [CHART_STRONG, CHART_WEAK]:
        if os.path.exists(chart):
            os.remove(chart)

def main():
    """Main execution"""
    print("Loading performance data...")
    strong_data = load_csv_data(STRONG_CSV, is_weak_scaling=False)
    weak_data = load_csv_data(WEAK_CSV, is_weak_scaling=True)
    
    if not strong_data:
        print(f"Error: No strong scaling data found in {STRONG_CSV}")
        return
    
    print("Generating charts...")
    generate_strong_scaling_chart(strong_data)
    generate_weak_scaling_chart(weak_data)
    
    print("Creating PDF report...")
    generate_pdf_report(strong_data, weak_data)
    
    print("\n=== Report generation complete ===")
    print(f"Output: {PDF_FILE}")

if __name__ == '__main__':
    main()
