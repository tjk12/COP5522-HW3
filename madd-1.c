#include <stdio.h>
#include <microtime.h>
#include <string.h>
#include <stdlib.h>


int main(int argc, char **argv)
{
  
  int N, i, j;
  double *A, *B, *C, result=0.0;
  double time1, time2;
  
  if(argc != 2)
  {
    fprintf(stderr, "USAGE: %s Matrix-Dimension\n", argv[0]);
    exit(1);
  }
  
  N = atoi(argv[1]);
  if(N <= 0 )
  {
    fprintf(stderr, "N = %d should be positive\n", N);
    exit(2);
  }    
  
  A = (void *) malloc(N*N*sizeof(A[0]));
  B = (void *) malloc(N*N*sizeof(B[0]));
  C = (void *) malloc(N*N*sizeof(C[0]));

  if(A==0 || B==0 || C==0)
    {
      fprintf(stderr, "Memory allocation failed in file %s, line %d\n", __FILE__, __LINE__);
      exit(1);
    }

  memset(C, 0, N*N*sizeof(C[0]));
  
  for(i=0; i<N; i++)
    for(j=0; j<N; j++)
      A[i*N+j] = B[i*N+j] = 0.1;
  
  time1 = microtime();

  for(i=0; i<N; i++)
      for(j=0; j<N; j++)
	C[i*N+j] += A[i*N+j] + B[i*N+j];

  time2 = microtime();
  
  printf("Time = %g us\tTimer Resolution = %g us\tPerformance = %g Gflop/s\n", time2-time1, get_microtime_resolution(), N*N*1e-3/(time2-time1));
  
  for(i=0; i<N; i++) /* Check correctness */
    for(j=0; j<N; j++)
      result += C[i*N+j];
  
  printf("Sum of matrix elements = %g\tExpected value = %g\n", result, (double) 0.2*N*N);
  
  return 0;
}
