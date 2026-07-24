#include <stdio.h>
#include <stdlib.h>
#include <stddef.h>

typedef struct {
  float *allocated;
  float *aligned;
  long long offset;
  long long sizes[2];
  long long strides[2];
} StridedMemRefType_f32_2D;

extern void _mlir_ciface_matrix_mul_trans_b(
  StridedMemRefType_f32_2D *out,
  StridedMemRefType_f32_2D *A,
  StridedMemRefType_f32_2D *B);
//golden standard C
static int matrix_mul_trans_B_pulp_open_fc(const float *src_a, const float *src_b, float *dst, const int dim_M, const int dim_N, const int dim_P)
{
    float sum;

    for (int m = 0; m < dim_M; m++) {
        for (int p = 0; p < dim_P; p++) {
            sum = 0;
            for (int n = 0; n < dim_N; n++)
                sum += src_a[m * dim_N + n] * src_b[p * dim_N + n];
            dst[m * dim_P + p] = sum;
        }
    }

    return 0;
}
static StridedMemRefType_f32_2D make_memref_2d(long long rows, long long cols) {
  StridedMemRefType_f32_2D mr;
  size_t elements = (size_t)(rows * cols);
  size_t bytes = elements * sizeof(float);
  float *data = (float*)aligned_alloc(64, bytes);
  mr.allocated = data;
  mr.aligned   = data;
  mr.offset    = 0;
  mr.sizes[0]   = rows;
  mr.sizes[1]   = cols;
  mr.strides[0] = cols;
  mr.strides[1] = 1;
  return mr;
}

static void free_memref(StridedMemRefType_f32_2D *mr) {
  free(mr->allocated);
  mr->allocated = mr->aligned = NULL;
}

static void random_fill(StridedMemRefType_f32_2D *mr, unsigned seed) {
  srand(seed);
  long long total_elements = mr->sizes[0] * mr->sizes[1];
  for (long long i = 0; i < total_elements; i++) {
    mr->aligned[mr->offset + i] = (float)((rand() % 2001) - 1000) / 100.0f;
  }
}

static double error_check(const StridedMemRefType_f32_2D *X,
                          const StridedMemRefType_f32_2D *Ref) {
  long long total_elements = X->sizes[0] * X->sizes[1];
  double s = 0.0;
  for (long long i = 0; i < total_elements; i++) {
      double dx = (double)X->aligned[X->offset + i]
                - (double)Ref->aligned[Ref->offset + i];
      s += dx * dx;
  }
  return s;
}

int main(int argc, char **argv) {
  long long M = 4;
  long long N = 4;
  long long P = 4;
  if (argc >= 4) {
      M = atoll(argv[1]);
      N = atoll(argv[2]);
      P = atoll(argv[3]);
  }

  StridedMemRefType_f32_2D A = make_memref_2d(M, N);
  StridedMemRefType_f32_2D B = make_memref_2d(P, N); 
  StridedMemRefType_f32_2D C = make_memref_2d(M, P);
  StridedMemRefType_f32_2D Cref = make_memref_2d(M, P);

  for (int i = 0; i <= 7; i++) {
      random_fill(&A, 13 + i);
      random_fill(&B, 17 + i);

      // Golden standard C
      matrix_mul_trans_B_pulp_open_fc(A.aligned, B.aligned, Cref.aligned, (int)M, (int)N, (int)P);

      // MLIR: out = matrix_mul_trans_b(A, B)
      _mlir_ciface_matrix_mul_trans_b(&C, &A, &B);

      double err = error_check(&C, &Cref);
      printf("Iterazione %d - Errore: %.6e\n", i, err);

      printf("Matrice MLIR (C):\n");
      for (long long m = 0; m < M; m++) {
        for (long long p = 0; p < P; p++) {
          printf("%7.2f ", C.aligned[m * P + p]);
        }
        printf("\n");
      }
      printf("Matrice C (Cref):\n");
      for (long long m = 0; m < M; m++) {
        for (long long p = 0; p < P; p++) {
          printf("%7.2f ", Cref.aligned[m * P + p]);
        }
        printf("\n");
      }
      printf("\n");
  }
  free_memref(&A);
  free_memref(&B);
  free_memref(&C);
  free_memref(&Cref);
  return 0;
}
