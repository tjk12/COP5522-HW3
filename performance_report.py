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
        plt.figure(figsize=(10, 6))
        
        sizes = df['size'].unique()
        for size in sizes:
            data = df[df['size'] == size]
            plt.plot(data['procs'], data['speedup'], 'o-', 
                    label=f'N = {size}', linewidth=2, markersize=8)
        
        # Plot ideal scaling
        max_procs = df['procs'].max()
        plt.plot([1, max_procs], [1, max_procs], 'k--', 
                label='Ideal', linewidth=1, alpha=0.5)
        
        plt.xlabel('Number of Cores', fontsize=12)
        plt.ylabel('Speedup', fontsize=12)
        plt.title('Strong Scaling Performance - Single Node', fontsize=14)
        plt.legend(fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(filename, dpi=300)
        plt.close()
        print(f"Saved plot: {filename}")
    
    def plot_weak_scaling(self, df, filename='weak_scaling_single_node.png'):
        """Generate weak scaling plot"""
        plt.figure(figsize=(10, 6))
        
        work_values = df['work_per_proc'].unique()
        for work in work_values:
            data = df[df['work_per_proc'] == work]
            plt.plot(data['procs'], data['efficiency'], 'o-',
                    label=f'Work/Proc = {work}', linewidth=2, markersize=8)
        
        # Plot ideal (100% efficiency)
        plt.axhline(y=100, color='k', linestyle='--', 
                   label='Ideal', linewidth=1, alpha=0.5)
        
        plt.xlabel('Number of Cores', fontsize=12)
        plt.ylabel('Parallel Efficiency (%)', fontsize=12)
        plt.title('Weak Scaling Performance - Single Node', fontsize=14)
        plt.legend(fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.ylim([0, 110])
        plt.tight_layout()
        plt.savefig(filename, dpi=300)
        plt.close()
        print(f"Saved plot: {filename}")
    
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
    
    # Run strong scaling tests
    print("\n" + "="*60)
    print("STRONG SCALING TESTS - SINGLE NODE")
    print("="*60)
    strong_df = analyzer.strong_scaling_single_node(matrix_sizes, proc_counts)
    
    if not strong_df.empty:
        analyzer.plot_strong_scaling(strong_df)
        analyzer.generate_report_table(strong_df, 'strong_scaling_results.csv')
        
        print("\n--- Strong Scaling Summary ---")
        print(strong_df.to_string(index=False))
    
    # Run weak scaling tests
    print("\n" + "="*60)
    print("WEAK SCALING TESTS - SINGLE NODE")
    print("="*60)
    weak_df = analyzer.weak_scaling_single_node(work_per_proc_values[0], proc_counts)
    
    if not weak_df.empty:
        analyzer.plot_weak_scaling(weak_df)
        analyzer.generate_report_table(weak_df, 'weak_scaling_results.csv')
        
        print("\n--- Weak Scaling Summary ---")
        print(weak_df.to_string(index=False))
    
    print("\n" + "="*60)
    print("Analysis Complete!")
    print("Check the generated PNG files and CSV tables")
    print("="*60)

if __name__ == "__main__":
    main()