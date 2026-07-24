
#include <stdio.h>
#include <stdlib.h>

typedef struct {
  float *allocated;
  float *aligned;
  long long offset;
  long long sizes[2];
  long long strides[2];
} StridedMemRefType_f32_2D;



static StridedMemRefType_f32_2D make_memref_2d(long long M, long long N) {
  StridedMemRefType_f32_2D mr;
  size_t bytes = (size_t)M * (size_t)N * sizeof(float);
  float *data = (float*)aligned_alloc(64, bytes);
  mr.allocated = data;
  mr.aligned   = data;
  mr.offset    = 0;
  mr.sizes[0]  = M;
  mr.sizes[1]  = N;
  mr.strides[0]= N;
  mr.strides[1]= 1;
  return mr;
}
extern void _mlir_ciface_matrix_memcpy(
  StridedMemRefType_f32_2D *out,
  StridedMemRefType_f32_2D *src);

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



// Golden Standard C
static int matrix_memcpy_pulp_open_fc(const float *src, float *dst, const int dim_M, const int dim_N)
{
    for (int m = 0; m < dim_M; m++)
        for (int n = 0; n < dim_N; n++)
            dst[m * dim_N + n] = src[m * dim_N + n];

    return 0;
}
// Error check (result VS golden model)
static double error_check (const StridedMemRefType_f32_2D *X,
                      const StridedMemRefType_f32_2D *Ref) {
  long long M = X->sizes[0], N = X->sizes[1];
  double s = 0.0;
  for (long long i=0;i<M;i++)
    for (long long j=0;j<N;j++) {
      double dx = (double)X->aligned[X->offset + i*X->strides[0] + j*X->strides[1]]
                - (double)Ref->aligned[Ref->offset + i*Ref->strides[0] + j*Ref->strides[1]];
      s += dx*dx;
    }
  return s;
}


int main(int argc, char **argv) {
    long long M = 4;
    long long N = 4;
    if (argc >= 3) {
      M = atoll(argv[1]);
      N = atoll(argv[2]);
    }

  StridedMemRefType_f32_2D Src = make_memref_2d(M, N);
  StridedMemRefType_f32_2D Dst = make_memref_2d(M, N);
  StridedMemRefType_f32_2D DstRef = make_memref_2d(M, N);

  for (int i = 0; i <= 6; i++) {
      random_fill(&Src, 42 + i);

      // Golden standard C
      matrix_memcpy_pulp_open_fc(Src.aligned, DstRef.aligned, (int)M, (int)N);

      // MLIR
      _mlir_ciface_matrix_memcpy(&Dst, &Src);

      double err = error_check(&Dst, &DstRef);
      printf("Iterazione %d - Errore: %.6e\n", i, err);

      printf("Matrice MLIR (Dst):\n");
      for (long long m = 0; m < M; m++) {
        for (long long n = 0; n < N; n++) {
          printf("%7.2f ", Dst.aligned[m * N + n]);
        }
        printf("\n");
      }
      printf("Matrice C (DstRef):\n");
      for (long long m = 0; m < M; m++) {
        for (long long n = 0; n < N; n++) {
          printf("%7.2f ", DstRef.aligned[m * N + n]);
        }
        printf("\n");
      }
      printf("\n");
  }
  free_memref(&Src);
  free_memref(&Dst);
  free_memref(&DstRef);
  return 0;
}