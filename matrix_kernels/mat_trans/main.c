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

extern void _mlir_ciface_mat_transpose(StridedMemRefType_f32_2D *out,
                                       StridedMemRefType_f32_2D *in);

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

// Libera la memoria
static void free_memref(StridedMemRefType_f32_2D *mr) {
  free(mr->allocated);
  mr->allocated = mr->aligned = NULL;
}

// Riempie la matrice 2D con valori casuali [-10, 10]
static void random_fill(StridedMemRefType_f32_2D *mr, unsigned seed) {
  srand(seed);
  long long total_elements = mr->sizes[0] * mr->sizes[1];
  for (long long i = 0; i < total_elements; i++) {
    mr->aligned[mr->offset + i] = (float)((rand() % 2001) - 1000) / 100.0f;
  }
}

// Riempie la matrice 2D con zeri
static void zero_fill(StridedMemRefType_f32_2D *mr) {
  long long total_elements = mr->sizes[0] * mr->sizes[1];
  for (long long i = 0; i < total_elements; i++) {
    mr->aligned[mr->offset + i] = 0.0f;
  }
}

// Golden Model
void ref_mat_transpose(float *src, float *dst, size_t dim_M, size_t dim_N) {
    for (size_t m = 0; m < dim_M; m++) {
        for (size_t n = 0; n < dim_N; n++) {
            dst[n * dim_M + m] = src[m * dim_N + n];
        }
    }
}

// Calcola l'errore tra il risultato MLIR e il Golden Model
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

static void print_matrix_corner(const char *name, const StridedMemRefType_f32_2D *mr, long long k) {
    // Stampa i k×k angoli: top-left, top-right, bottom-left, bottom-right
    long long R = mr->sizes[0], C = mr->sizes[1];
    long long kr = (k < R) ? k : R;
    long long kc = (k < C) ? k : C;

    printf("\n=== %s (%lld x %lld) — angoli %lld×%lld ===\n", name, R, C, kr, kc);

    printf("[top-left]\n");
    for (long long r = 0; r < kr; r++) {
        for (long long c = 0; c < kc; c++)
            printf("%7.2f ", mr->aligned[mr->offset + r*mr->strides[0] + c*mr->strides[1]]);
        printf("\n");
    }
    // printf("[top-right]\n");
    // for (long long r = 0; r < kr; r++) {
    //     for (long long c = C-kc; c < C; c++)
    //         printf("%7.2f ", mr->aligned[mr->offset + r*mr->strides[0] + c*mr->strides[1]]);
    //     printf("\n");
    // }
    // printf("[bottom-left]\n");
    // for (long long r = R-kr; r < R; r++) {
    //     for (long long c = 0; c < kc; c++)
    //         printf("%7.2f ", mr->aligned[mr->offset + r*mr->strides[0] + c*mr->strides[1]]);
    //     printf("\n");
    // }
    // printf("[bottom-right]\n");
    // for (long long r = R-kr; r < R; r++) {
    //     for (long long c = C-kc; c < C; c++)
    //         printf("%7.2f ", mr->aligned[mr->offset + r*mr->strides[0] + c*mr->strides[1]]);
    //     printf("\n");
    // }
}

int main(int argc, char **argv) {
  long long M = 64; 
  long long N = 64; 
  if (argc == 3) { 
      M = atoll(argv[1]); 
      N = atoll(argv[2]); 
  }

  StridedMemRefType_f32_2D A = make_memref_2d(M, N);
  StridedMemRefType_f32_2D B = make_memref_2d(M, N); 
  StridedMemRefType_f32_2D C = make_memref_2d(N, M);
  StridedMemRefType_f32_2D Cref = make_memref_2d(N, M);

  for (long long i = 0; i <= 6; i++) {
        random_fill(&A, 13 + i); 
        random_fill(&B, 42 + i);
        zero_fill(&C);
        zero_fill(&Cref);

        // Esecuzione MLIR 
        _mlir_ciface_mat_transpose(&C, &A);
        
        // Esecuzione Golden Standard in C 
        ref_mat_transpose(A.aligned, Cref.aligned, M, N);
        print_matrix_corner("A (input)", &A, 4);
        print_matrix_corner("C - MLIR transpose", &C, 4);
        print_matrix_corner("Cref - Golden", &Cref, 4);
        // Controllo dell'errore
        double err = error_check(&C, &Cref);
        printf("Iterazione %lld - Error: %.6e\n", i, err);
  }

  free_memref(&A); 
  free_memref(&B); 
  free_memref(&C); 
  free_memref(&Cref);
  
  return 0;
}