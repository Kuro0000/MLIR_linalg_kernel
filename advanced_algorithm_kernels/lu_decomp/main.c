#include <stdio.h>
#include <stdlib.h>
#include <stddef.h>
#include <math.h>


typedef struct {
  float *allocated;
  float *aligned;
  long long offset;
  long long sizes[2];
  long long strides[2];
} StridedMemRefType_f32_2D;

typedef struct {
  int *allocated;
  int *aligned;
  long long offset;
  long long sizes[1];
  long long strides[1];
} StridedMemRefType_i32_1D;


typedef struct {
    StridedMemRefType_f32_2D mat;
    StridedMemRefType_i32_1D perm;
} LUResult;

extern void _mlir_ciface_linalg_lu_decomp(
    LUResult *out, 
    StridedMemRefType_f32_2D *mat_in
);


static StridedMemRefType_f32_2D make_memref_2d(long long rows, long long cols) {
  StridedMemRefType_f32_2D mr;
  size_t bytes = (size_t)(rows * cols) * sizeof(float);
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

static void free_memref_2d(StridedMemRefType_f32_2D *mr) {
  if(mr->allocated) free(mr->allocated);
  mr->allocated = mr->aligned = NULL;
}

static void free_memref_1d_i32(StridedMemRefType_i32_1D *mr) {
  if(mr->allocated) free(mr->allocated);
  mr->allocated = mr->aligned = NULL;
}

static void random_fill(StridedMemRefType_f32_2D *mr, unsigned seed) {
  srand(seed);
  long long total_elements = mr->sizes[0] * mr->sizes[1];
  for (long long i = 0; i < total_elements; i++) {
    // Valori tra -10.0 e 10.0
    mr->aligned[mr->offset + i] = (float)((rand() % 2001) - 1000) / 100.0f;
  }
}

static void copy_memref(const StridedMemRefType_f32_2D *src, StridedMemRefType_f32_2D *dst) {
  long long total_elements = src->sizes[0] * src->sizes[1];
  for (long long i = 0; i < total_elements; i++) {
    dst->aligned[dst->offset + i] = src->aligned[src->offset + i];
  }
}

//goldenStandard
static int set_permutation_identity(int *vec, const int len) {
    for (int i = 0; i < len; i++)
        vec[i] = i;
    return 0;
}

static void swap_rows(float *mat, int row_a, int row_b, int dim_M, int dim_N) {
    for (int n = 0; n < dim_N; n++) {
        float tmp = mat[row_a * dim_N + n];
        mat[row_a * dim_N + n] = mat[row_b * dim_N + n];
        mat[row_b * dim_N + n] = tmp;
    }
}

static void swap_elems(int *vec, int a, int b) {
    int tmp = vec[a];
    vec[a] = vec[b];
    vec[b] = tmp;
}

static int linalg_lu_decomp_pulp_open_fc(float *mat, int *perm, const int dim_M, const int dim_N)
{
    float factor;
    float pivot;
    int dim_min;
    int row_max;
    float max;
    float val;

    set_permutation_identity(perm, dim_M);

    dim_min = dim_M < dim_N ? dim_M : dim_N;
    for (int k = 0; k < dim_min; k++) {

        /* Partial Pivoting */
        row_max = k;
        max = fabs(mat[k * dim_N + k]);
        for (int m = k + 1; m < dim_M; m++) {
            val = fabs(mat[m * dim_N + k]);
            if (val > max) {
                row_max = m;
                max = val;
            }
        }

        if (row_max != k) {
            swap_rows(mat, k, row_max, dim_M, dim_N);
            swap_elems(perm, k, row_max);
        }

        /* Gaussian Elimination */
        pivot = mat[k * dim_N + k];
        if (pivot == 0.0f) {
            printf("ERROR | Zero pivot found at position %d - Matrix is singular\n", k);
            exit(-1);
        }

        for (int m = k + 1; m < dim_M; m++) {
            factor = mat[m * dim_N + k] / pivot;
            mat[m * dim_N + k] = factor;

            for (int n = k + 1; n < dim_N; n++)
                mat[m * dim_N + n] -= factor * mat[k * dim_N + n];
        }
    }

    return 0;
}

static double error_check_mat(const StridedMemRefType_f32_2D *X, const float *Ref) {
  long long total_elements = X->sizes[0] * X->sizes[1];
  double s = 0.0;
  for (long long i = 0; i < total_elements; i++) {
      double dx = (double)X->aligned[X->offset + i] - (double)Ref[i];
      s += dx * dx;
  }
  return s;
}

static int error_check_perm(const StridedMemRefType_i32_1D *X, const int *Ref) {
  long long total_elements = X->sizes[0];
  int mismatches = 0;
  for (long long i = 0; i < total_elements; i++) {
      if (X->aligned[X->offset + i] != Ref[i]) {
          mismatches++;
      }
  }
  return mismatches;
}

static void print_matrix_corner(const char *name, const float *mat, long long R, long long C, long long k) {
    long long kr = (k < R) ? k : R;
    long long kc = (k < C) ? k : C;
    printf("\n=== %s (%lld x %lld) ===\n", name, R, C);
    for (long long r = 0; r < kr; r++) {
        for (long long c = 0; c < kc; c++)
            printf("%7.2f ", mat[r * C + c]);
        printf("\n");
    }
}
//main
int main(int argc, char **argv) {
    long long M = 4, N = 4;
    if (argc >= 3) {
        M = atoll(argv[1]); 
        N = atoll(argv[2]);
    }

    StridedMemRefType_f32_2D MatIn  = make_memref_2d(M, N);
    StridedMemRefType_f32_2D MatRef = make_memref_2d(M, N);
    
    int *perm_ref = (int*)malloc(M * sizeof(int));

    for (long long i = 1; i <= 5; i++) {


        random_fill(&MatIn, 42 + i);
        copy_memref(&MatIn, &MatRef); 
        LUResult mlir_res = {0}; 
        _mlir_ciface_linalg_lu_decomp(&mlir_res, &MatIn);

        linalg_lu_decomp_pulp_open_fc(MatRef.aligned, perm_ref, (int)M, (int)N);

        double err_mat = error_check_mat(&mlir_res.mat, MatRef.aligned);
        int err_perm   = error_check_perm(&mlir_res.perm, perm_ref);

        printf("  Errore Matrice LU (Sum of Sq) : %.6e\n", err_mat);
        printf("  Mismatches Array Permutazioni : %d\n", err_perm);

        if (err_mat < 1e-3 && err_perm == 0) {
            printf("  Esito: SUPERATO, Errore = %e\n", err_mat);
        } else {
            printf("  Esito: FALLITO, Errore = %e\n", err_mat);

            print_matrix_corner("MLIR Output", mlir_res.mat.aligned, M, N, 4);
            print_matrix_corner("Golden C Reference", MatRef.aligned, M, N, 4);
            
            printf("\nPermutazioni MLIR: ");
            for(int p=0; p<M; p++) printf("%d ", mlir_res.perm.aligned[p]);
            printf("\nPermutazioni C   : ");
            for(int p=0; p<M; p++) printf("%d ", perm_ref[p]);
            printf("\n");
        }

        free_memref_2d(&mlir_res.mat);
        free_memref_1d_i32(&mlir_res.perm);
    }

    free_memref_2d(&MatIn);
    free_memref_2d(&MatRef);
    free(perm_ref);

    return 0;
}
