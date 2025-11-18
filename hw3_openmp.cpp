#include <omp.h>
#include <iostream>
#include <fstream>
#include <cstring>
#include <cstdlib>
#include <chrono>

using namespace std;

typedef float* Matrix;

Matrix CreateMatrix(int size) {
    return new float[size];
}

void FreeMatrix(Matrix M) {
    if (M) delete[] M;
}

void InitMatrix(Matrix A, int Rows, int Cols) {
    for (int i = 0; i < Rows; i++)
        for (int j = 0; j < Cols; j++)
            A[i * Cols + j] = 1.0f / (i + j + 2);
}

// OpenMP matrix-vector multiplication
void MatVecMult(Matrix A, Matrix B, Matrix C, int ARows, int ACols) {
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < ARows; i++) {
        float sum = 0.0f;
        const float* row = &A[i * ACols];
        
        // Manual unrolling for better performance
        int j = 0;
        for (; j <= ACols - 4; j += 4) {
            sum += row[j] * B[j];
            sum += row[j+1] * B[j+1];
            sum += row[j+2] * B[j+2];
            sum += row[j+3] * B[j+3];
        }
        for (; j < ACols; j++) {
            sum += row[j] * B[j];
        }
        C[i] = sum;
    }
}

int main(int argc, char** argv) {
    int N, M;
    Matrix A, B, C;
    
    // Read number of threads from command line or use default
    int num_threads = (argc > 1) ? atoi(argv[1]) : omp_get_max_threads();
    omp_set_num_threads(num_threads);
    
    // Read matrix size from input.txt
    ifstream infile("input.txt");
    if (!infile) {
        cerr << "Error: Cannot open input.txt" << endl;
        return 1;
    }
    infile >> N;
    infile.close();
    
    if (N <= 0) {
        cerr << "Error: Invalid matrix size" << endl;
        return 1;
    }
    
    M = N;
    
    // Allocate matrices
    A = CreateMatrix(N * N);
    B = CreateMatrix(N);
    C = CreateMatrix(N);
    
    // Initialize matrices
    InitMatrix(A, N, N);
    InitMatrix(B, N, 1);
    memset(C, 0, N * sizeof(float));
    
    // Timing
    auto start = chrono::high_resolution_clock::now();
    
    MatVecMult(A, B, C, N, N);
    
    auto end = chrono::high_resolution_clock::now();
    auto duration = chrono::duration_cast<chrono::microseconds>(end - start);
    double time_us = duration.count();
    
    // Calculate performance
    double flops = 2.0 * N * N;  // N^2 multiplies and N^2 adds
    double gflops = (flops / time_us) * 1e-3;  // Convert to Gflop/s
    
    // Print results
    cout << "\nMatrix Size: " << N << "x" << N << endl;
    cout << "Number of Threads: " << num_threads << endl;
    cout << "Time = " << time_us << " us" << endl;
    cout << "Performance = " << gflops << " Gflop/s" << endl;
    cout << "C[N/2] = " << C[N/2] << endl;
    
    // Cleanup
    FreeMatrix(A);
    FreeMatrix(B);
    FreeMatrix(C);
    
    return 0;
}
