#include <stdio.h>
#include <stdlib.h>
#include <time.h> // FONDAMENTALE per clock_gettime

typedef struct {
  float *allocated;
  float *aligned;
  long long offset;
  long long sizes[2];
  long long strides[2];
} StridedMemRefType_f32_2D;

extern void _mlir_ciface_matrix_matmul(StridedMemRefType_f32_2D *out,
                                       StridedMemRefType_f32_2D *A,
                                       StridedMemRefType_f32_2D *B);

static StridedMemRefType_f32_2D make_memref_2d(long long M, long long N) {
  StridedMemRefType_f32_2D mr;
  size_t bytes = (size_t)M * (size_t)N * sizeof(float);
  float *data = (float*)aligned_alloc(64, (bytes + 63) & ~63);
  mr.allocated = data;
  mr.aligned   = data;
  mr.offset    = 0;
  mr.sizes[0]  = M;
  mr.sizes[1]  = N;
  mr.strides[0]= N;
  mr.strides[1]= 1;
  return mr;
}

static void free_memref(StridedMemRefType_f32_2D *mr) {
  free(mr->allocated);
  mr->allocated = mr->aligned = NULL;
}

static void random_fill(StridedMemRefType_f32_2D *mr, unsigned seed) {
  srand(seed);
  long long M = mr->sizes[0], N = mr->sizes[1];
  for (long long i=0;i<M;i++) {
    for (long long j=0;j<N;j++) {
      mr->aligned[mr->offset + i*mr->strides[0] + j*mr->strides[1]] =
      (float)((rand() % 2001) - 1000) / 100.0f;
    }
  }
}

static void zero_fill(StridedMemRefType_f32_2D *mr) {
  long long M = mr->sizes[0], N = mr->sizes[1];
  for (long long i=0;i<M;i++)
    for (long long j=0;j<N;j++)
      mr->aligned[mr->offset + i*mr->strides[0] + j*mr->strides[1]] = 0.0f;
}

double get_time_diff(struct timespec start, struct timespec end) {
  return (double)(end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;
}

// Golden model
static int matrix_mul_pulp_open_fc(const float *src_a, const float *src_b, float *dst, const int dim_M, const int dim_N, const int dim_P) {
  float sum;
  for (int m = 0; m < dim_M; m++) {
    for (int p = 0; p < dim_P; p++) {
      sum = 0;
      for (int n = 0; n < dim_N; n++)
        sum += src_a[m * dim_N + n] * src_b[n * dim_P + p];
      dst[m * dim_P + p] = sum;
    }
  }
  return 0;
}

// Error check
static double error_check (const StridedMemRefType_f32_2D *X, const StridedMemRefType_f32_2D *Ref) {
  long long M = X->sizes[0], N = X->sizes[1];
  double s = 0.0;
  for (long long i=0;i<M;i++)
    for (long long j=0;j<N;j++) {
      double dx = (double)X->aligned[X->offset + i*X->strides[0] + j*X->strides[1]]
      - (double)Ref->aligned[Ref->offset + i*Ref->strides[0] + j*Ref->strides[1]];
      s += dx*dx;
    }
    return s / (M * N); // Errore medio
}

int main(int argc, char **argv) {
  // AUMENTIAMO LA DIMENSIONE DI DEFAULT! 64x64 è troppo piccolo per misurare il parallelismo.
  long long M = 1024, K = 1024, N = 1024;
  if (argc == 4) { M = atoll(argv[1]); K = atoll(argv[2]); N = atoll(argv[3]); }

  // Rileva i thread dall'ambiente (default 1)
  char *env_threads = getenv("OMP_NUM_THREADS");
  int threads = env_threads ? atoi(env_threads) : 1;

  printf("\n=== MatMul Benchmark (%lldx%lldx%lld) ===\n", M, K, N);
  printf("Threads in uso : %d\n", threads);

  StridedMemRefType_f32_2D A = make_memref_2d(M, K);
  StridedMemRefType_f32_2D B = make_memref_2d(K, N);
  StridedMemRefType_f32_2D C = make_memref_2d(M, N);
  StridedMemRefType_f32_2D Cref = make_memref_2d(M, N);

  random_fill(&A, 13);
  random_fill(&B, 42);
  zero_fill(&C);
  zero_fill(&Cref);

  // ---------------------------------------------------------
  // 1. FASE DI VERIFICA (Eseguita una sola volta)
  // ---------------------------------------------------------
  printf("Verifica correttezza in corso...\n");
  _mlir_ciface_matrix_matmul(&C, &A, &B);
  matrix_mul_pulp_open_fc(A.aligned, B.aligned, Cref.aligned, (int)M, (int)K, (int)N);

  double err = error_check(&C, &Cref);
  if (err > 1e-3) {
    printf("Errore matematico eccessivo: %.6e. Interruzione.\n", err);
    return 1;
  }
  printf("Esito: SUCCESSO (Errore = %.6e)\n", err);

  // ---------------------------------------------------------
  // 2. FASE DI BENCHMARK (Solo MLIR)
  // ---------------------------------------------------------
  // ---------------------------------------------------------
  // 2. FASE DI BENCHMARK (Test Singoli e Progressivi)
  // ---------------------------------------------------------
  int num_runs = 5;

  printf("Esecuzione Benchmark (%d test singoli)...\n", num_runs);
  printf("----------------------------------------\n");
  for(int i = 1; i <= num_runs; i++) {
    zero_fill(&C); // Resetta l'output ad ogni ciclo

    struct timespec start, end;
    clock_gettime(CLOCK_MONOTONIC, &start);

    _mlir_ciface_matrix_matmul(&C, &A, &B);

    clock_gettime(CLOCK_MONOTONIC, &end);

    double iter_time = get_time_diff(start, end);
    printf("Test %d: %.4f secondi\n", i, iter_time);
  }
  printf("----------------------------------------\n");

  free_memref(&A); free_memref(&B); free_memref(&C); free_memref(&Cref);
  return 0;
}
