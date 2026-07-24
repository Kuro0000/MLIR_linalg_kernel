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

extern void _mlir_ciface_mat_swaprow(StridedMemRefType_f32_2D *result,
                                      StridedMemRefType_f32_2D *mat,
                                      long long row_a, long long row_b);

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

// Copia una matrice
static void copy_memref(const StridedMemRefType_f32_2D *src, StridedMemRefType_f32_2D *dst) {
  long long total_elements = src->sizes[0] * src->sizes[1];
  for (long long i = 0; i < total_elements; i++) {
    dst->aligned[dst->offset + i] = src->aligned[src->offset + i];
  }
}

// Golden Model (swap rows)
void ref_mat_swaprow(float *mat, long long row_a, long long row_b, long long dim_N) {
    for (long long n = 0; n < dim_N; n++) {
        float tmp = mat[row_a * dim_N + n];
        mat[row_a * dim_N + n] = mat[row_b * dim_N + n];
        mat[row_b * dim_N + n] = tmp;
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
    long long M = 4, N = 4, row_a = 3, row_b = 2;
    if (argc >= 5) {
        M = atoll(argv[1]); N = atoll(argv[2]);
        row_a = atoll(argv[3]); row_b = atoll(argv[4]);
    }

    StridedMemRefType_f32_2D Mat    = make_memref_2d(M, N);
    StridedMemRefType_f32_2D MatRef = make_memref_2d(M, N);
    StridedMemRefType_f32_2D MatOut = make_memref_2d(M, N);

    for (long long i = 0; i <= 6; i++) {
        random_fill(&Mat, 13 + i);
        copy_memref(&Mat, &MatRef);

        // Golden model 
        ref_mat_swaprow(MatRef.aligned, row_a, row_b, N);

        // MLIR: legge Mat, scrive in MatOut
        _mlir_ciface_mat_swaprow(&MatOut, &Mat, row_a, row_b);

        printf("""\nIterazione %lld - Swap righe %lld e %lld\n", i, row_a, row_b);
        print_matrix_corner("Input", &Mat, 4);
        print_matrix_corner("MLIR Output", &MatOut, 4);
        print_matrix_corner("Reference", &MatRef, 4);
        double err = error_check(&MatOut, &MatRef);
        printf("Iterazione %lld - Error: %.6e\n", i, err);
    }

    free_memref(&Mat);
    free_memref(&MatRef);
    free_memref(&MatOut);
    return 0;
}