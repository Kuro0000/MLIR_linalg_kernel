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


extern void _mlir_ciface_mat_setall(StridedMemRefType_f32_2D *out, 
                                  float val, 
                                  StridedMemRefType_f32_2D *init);
// Golden Standard C
static int matrix_set_all_pulp_open_fc(float *mat, float val, int dim_M, int dim_N) {
    for (int m = 0; m < dim_M; m++)
        for (int n = 0; n < dim_N; n++)
            mat[m * dim_N + n] = val;
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
  float val;
  if (argc >= 4) {
      M = atoll(argv[1]);
      N = atoll(argv[2]);
      val = atof(argv[3]);
  }

    // Sagoma di input (tensor vuoto)
    StridedMemRefType_f32_2D Init = make_memref_2d(M, N);
    StridedMemRefType_f32_2D Out = make_memref_2d(M, N);
    StridedMemRefType_f32_2D Ref = make_memref_2d(M, N);
  for(int i = 0; i <= 6; i++) {
      // Golden standard C
      val =(float)((rand() % 2001) - 1000) / 100.0f;
      matrix_set_all_pulp_open_fc(Ref.aligned, val, (int)M, (int)N);

      // MLIR 
      _mlir_ciface_mat_setall(&Out, val, &Init);

      double err = error_check(&Out, &Ref);
      printf("Errore: %.6e\n", err);

      printf("Matrice MLIR (Out):\n");
      for (long long m = 0; m < M; m++) {
        for (long long n = 0; n < N; n++) {
          printf("%7.2f ", Out.aligned[m * N + n]);
        }
        printf("\n");
      }
      printf("Matrice C (Ref):\n");
      for (long long m = 0; m < M; m++) {
        for (long long n = 0; n < N; n++) {
          printf("%7.2f ", Ref.aligned[m * N + n]);
        }
        printf("\n");
      }
    }
    free_memref(&Init);
    free_memref(&Out);
    free_memref(&Ref);
    return 0;
}
