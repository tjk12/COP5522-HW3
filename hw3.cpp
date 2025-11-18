#include <mpi.h>
#include <iostream>
#include <fstream>
#include <cstring>
#include <cstdlib>
#include <chrono>
#include <vector>
#include <algorithm>

#if defined(__AVX2__)
#include <immintrin.h>
#elif defined(__ARM_NEON)
#include <arm_neon.h>
#endif

using namespace std;

double microtime() {
    auto now = chrono::high_resolution_clock::now();
    auto duration = now.time_since_epoch();
    return chrono::duration_cast<chrono::microseconds>(duration).count();
}

typedef float* Matrix;

Matrix CreateMatrix(int size) {
    Matrix M = new(nothrow) float[size];
    if (M == nullptr) {
        cerr << "Matrix allocation failed" << endl;
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    return M;
}

void FreeMatrix(Matrix M) {
    if (M) delete[] M;
}

void InitMatrix(Matrix A, int Rows, int Cols) {
    for (int i = 0; i < Rows; i++)
        for (int j = 0; j < Cols; j++)
            A[i * Cols + j] = 1.0f / (i + j + 2);
}

// Optimized matrix-vector multiplication for local rows
// Prioritizes AVX2/FMA path for Linux x86-64 systems
void MatVecMultLocal(Matrix A, Matrix B, Matrix C, int ARows, int ACols) {
    // Compute local matrix-vector product (row-major, cache-friendly)
#if defined(__AVX2__)
    for (int i = 0; i < ARows; ++i) {
        const float *row = &A[i * ACols];
        __m256 vsum0 = _mm256_setzero_ps();
        __m256 vsum1 = _mm256_setzero_ps();

        int k = 0;
        int limit = ACols - (ACols % 16);
        for (; k < limit; k += 16) {
            __m256 va0 = _mm256_loadu_ps(row + k);
            __m256 vb0 = _mm256_loadu_ps(B + k);
#if defined(__FMA__)
            vsum0 = _mm256_fmadd_ps(va0, vb0, vsum0);
#else
            vsum0 = _mm256_add_ps(vsum0, _mm256_mul_ps(va0, vb0));
#endif

            __m256 va1 = _mm256_loadu_ps(row + k + 8);
            __m256 vb1 = _mm256_loadu_ps(B + k + 8);
#if defined(__FMA__)
            vsum1 = _mm256_fmadd_ps(va1, vb1, vsum1);
#else
            vsum1 = _mm256_add_ps(vsum1, _mm256_mul_ps(va1, vb1));
#endif
        }

        __m256 vsum = _mm256_add_ps(vsum0, vsum1);

        for (; k <= ACols - 8; k += 8) {
            __m256 va = _mm256_loadu_ps(row + k);
            __m256 vb = _mm256_loadu_ps(B + k);
#if defined(__FMA__)
            vsum = _mm256_fmadd_ps(va, vb, vsum);
#else
            vsum = _mm256_add_ps(vsum, _mm256_mul_ps(va, vb));
#endif
        }

        __m128 low = _mm256_castps256_ps128(vsum);
        __m128 high = _mm256_extractf128_ps(vsum, 1);
        __m128 sum128 = _mm_add_ps(low, high);
        sum128 = _mm_hadd_ps(sum128, sum128);
        sum128 = _mm_hadd_ps(sum128, sum128);
        float sum0 = _mm_cvtss_f32(sum128);

        for (; k < ACols; ++k) {
            sum0 += row[k] * B[k];
        }

        C[i] = sum0;
    }
#elif defined(__ARM_NEON)
    for (int i = 0; i < ARows; ++i) {
        const float *row = &A[i * ACols];
        float32x4_t vsum0 = vdupq_n_f32(0.0f);
        float32x4_t vsum1 = vdupq_n_f32(0.0f);

        int k = 0;
        int limit = ACols - (ACols % 8);
        for (; k < limit; k += 8) {
            float32x4_t va0 = vld1q_f32(row + k);
            float32x4_t vb0 = vld1q_f32(B + k);
            vsum0 = vmlaq_f32(vsum0, va0, vb0);

            float32x4_t va1 = vld1q_f32(row + k + 4);
            float32x4_t vb1 = vld1q_f32(B + k + 4);
            vsum1 = vmlaq_f32(vsum1, va1, vb1);
        }

        float32x4_t vsum = vaddq_f32(vsum0, vsum1);
        float sum0 = vaddvq_f32(vsum);

        for (; k < ACols; ++k) {
            sum0 += row[k] * B[k];
        }

        C[i] = sum0;
    }
#else
    for (int i = 0; i < ARows; ++i) {
        const float *row = &A[i * ACols];
        float sum0 = 0.0f;

        int k = 0;
        int limit = ACols - (ACols % 8);
        float sum1 = 0.0f, sum2 = 0.0f, sum3 = 0.0f;
        for (; k < limit; k += 8) {
            sum0 += row[k] * B[k];
            sum1 += row[k + 1] * B[k + 1];
            sum2 += row[k + 2] * B[k + 2];
            sum3 += row[k + 3] * B[k + 3];
            sum0 += row[k + 4] * B[k + 4];
            sum1 += row[k + 5] * B[k + 5];
            sum2 += row[k + 6] * B[k + 6];
            sum3 += row[k + 7] * B[k + 7];
        }

        sum0 += sum1 + sum2 + sum3;

        for (; k < ACols; ++k) {
            sum0 += row[k] * B[k];
        }

        C[i] = sum0;
    }
#endif
}

int main(int argc, char **argv) {
    int rank, size;
    int N, M, P = 1;
    int expected_procs = 0;
    Matrix A_local, B, C_local, C_global = nullptr;
    double time_start, time_end;
    
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    
    // Check for command line argument specifying number of processes
    if (argc > 1) {
        expected_procs = atoi(argv[1]);
        if (rank == 0 && expected_procs != size) {
            cerr << "Warning: Expected " << expected_procs << " processes (from command line), but running with " << size << " processes" << endl;
        }
    }
    
    // Create user-defined datatype for matrix size (practice as required)
    MPI_Datatype matrix_size_type;
    MPI_Type_contiguous(1, MPI_INT, &matrix_size_type);
    MPI_Type_commit(&matrix_size_type);
    
    // Process 0 reads matrix size from file
    if (rank == 0) {
        ifstream infile("input.txt");
        if (!infile) {
            cerr << "Error: Cannot open input.txt" << endl;
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        infile >> N;
        infile.close();
        
        if (N <= 0) {
            cerr << "Error: Invalid matrix size" << endl;
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        
        // Send matrix size to all other processes using point-to-point
        for (int i = 1; i < size; i++) {
            MPI_Send(&N, 1, matrix_size_type, i, 0, MPI_COMM_WORLD);
        }
    } else {
        // Other processes receive matrix size
        MPI_Recv(&N, 1, matrix_size_type, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }
    
    M = N;
    
    // Calculate rows per process
    int rows_per_proc = N / size;
    int remainder = N % size;
    int local_rows = rows_per_proc + (rank < remainder ? 1 : 0);
    
    // Calculate displacement for this process
    int offset = rank * rows_per_proc + min(rank, remainder);
    
    // Allocate local matrix portion and vector
    A_local = CreateMatrix(local_rows * M);
    B = CreateMatrix(M);
    C_local = CreateMatrix(local_rows);
    
    // Initialize local portion of matrix A (row-major)
    for (int i = 0; i < local_rows; i++) {
        int global_row = offset + i;
        for (int j = 0; j < M; j++) {
            A_local[i * M + j] = 1.0f / (global_row + j + 2);
        }
    }
    
    // Initialize vector B (same on all processes)
    InitMatrix(B, M, P);
    
    // Barrier to ensure all processes are ready
    MPI_Barrier(MPI_COMM_WORLD);
    
    // Start timing (excluding file I/O and initial communication)
    time_start = microtime();
    
    // Perform local matrix-vector multiplication
    MatVecMultLocal(A_local, B, C_local, local_rows, M);
    
    // Gather results at process 0
    if (rank == 0) {
        C_global = CreateMatrix(N);
    }
    
    // Prepare receive counts and displacements for Gatherv
    vector<int> recvcounts(size);
    vector<int> displs(size);
    
    if (rank == 0) {
        for (int i = 0; i < size; i++) {
            recvcounts[i] = rows_per_proc + (i < remainder ? 1 : 0);
            displs[i] = i * rows_per_proc + min(i, remainder);
        }
    }
    
    MPI_Gatherv(C_local, local_rows, MPI_FLOAT,
                C_global, recvcounts.data(), displs.data(), MPI_FLOAT,
                0, MPI_COMM_WORLD);
    
    // End timing
    time_end = microtime();
    
    // Process 0 prints results
    if (rank == 0) {
        double elapsed = time_end - time_start;
        double gflops = 2.0 * N * N * 1e-3 / elapsed;
        
        cout << "\nMatrix Size: " << N << "x" << N << endl;
        cout << "Number of Processes: " << size << endl;
        cout << "Time = " << elapsed << " us" << endl;
        cout << "Performance = " << gflops << " Gflop/s" << endl;
        cout << "C[N/2] = " << static_cast<double>(C_global[N/2]) << endl << endl;
    }
    
    // Cleanup
    FreeMatrix(A_local);
    FreeMatrix(B);
    FreeMatrix(C_local);
    if (rank == 0) FreeMatrix(C_global);
    
    MPI_Type_free(&matrix_size_type);
    MPI_Finalize();
    
    return 0;
}