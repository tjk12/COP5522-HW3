// Mv.cpp
#include <iostream>
#include <chrono>
#include <cstring>
#include <cstdlib>

using namespace std;

double microtime() {
    auto now = chrono::high_resolution_clock::now();
    auto duration = now.time_since_epoch();
    return chrono::duration_cast<chrono::microseconds>(duration).count();
}

double get_microtime_resolution() {
    return 1.0; // 1 microsecond resolution for chrono::microseconds
}

typedef float* Matrix;

Matrix CreateMatrix(int Rows, int Cols)
{
    Matrix M;
    
    M = new(nothrow) float[Rows * Cols];
    if (M == nullptr)
        cerr << "Matrix allocation failed in file " << __FILE__ << ", line " << __LINE__ << endl;
    
    return M;
}

void FreeMatrix(Matrix M)
{
    if (M)
        delete[] M;
}

void InitMatrix(Matrix A, int Rows, int Cols)
{
    int i, j;
    
    for (i = 0; i < Rows; i++)
        for (j = 0; j < Cols; j++)
            A[i * Cols + j] = 1.0f / (i + j + 2);
}

void MatVecMult(Matrix A, Matrix B, Matrix C, int ARows, int ACols)
{
    int i, k;

    // Zero output
    memset(C, 0, ARows * sizeof(C[0]));

    // Optimized: iterate over rows outer, columns inner for cache-friendly access
    for (i = 0; i < ARows; ++i) {
        const float *row = &A[i * ACols];
        float sum0 = 0.0f;

        int k = 0;
        int limit = ACols - (ACols % 4);
        // Unroll by 4
        for (; k < limit; k += 4) {
            sum0 += row[k] * B[k];
            sum0 += row[k+1] * B[k+1];
            sum0 += row[k+2] * B[k+2];
            sum0 += row[k+3] * B[k+3];
        }
        // Remainder
        for (; k < ACols; ++k) {
            sum0 += row[k] * B[k];
        }

        C[i] = sum0;
    }
}

int main(int argc, char **argv)
{
    int N, M, P = 1;
    Matrix A, B, C;
    double t, time1, time2;
    
    if (argc != 2)
    {
        cerr << "USAGE: " << argv[0] << " Matrix-Dimension" << endl;
        exit(1);
    }
    
    N = atoi(argv[1]);
    M = N;
    
    A = CreateMatrix(N, M);
    B = CreateMatrix(M, P);
    C = CreateMatrix(N, P);
    
    InitMatrix(A, N, M);
    InitMatrix(B, M, P);
    memset(C, 0, N * P * sizeof(C[0]));
    
    time1 = microtime();
    MatVecMult(A, B, C, N, M);
    time2 = microtime();
    
    t = time2 - time1;
    cout << "\nTime = " << t << " us\tTimer Resolution = " << get_microtime_resolution() 
         << " us\tPerformance = " << 2.0 * N * N * 1e-3 / t << " Gflop/s" << endl;
    cout << "C[N/2] = " << static_cast<double>(C[N/2]) << "\n" << endl;
    
    FreeMatrix(A);
    FreeMatrix(B);
    FreeMatrix(C);
    
    return 0;
}