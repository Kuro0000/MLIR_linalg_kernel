#include <stdio.h>
#include <stdlib.h>
#include <math.h>


typedef struct {
    float *allocated; float *aligned; long long offset;
    long long sizes[2]; long long strides[2];
} StridedMemRefType_f32_2D;

typedef struct {
    float *allocated; float *aligned; long long offset;
    long long sizes[1]; long long strides[1];
} StridedMemRefType_f32_1D;

typedef struct {
    int *allocated; int *aligned; long long offset;
    long long sizes[1]; long long strides[1];
} StridedMemRefType_i32_1D;

extern void _mlir_ciface_linalg_lu_solve(
    StridedMemRefType_f32_1D *out,
    StridedMemRefType_f32_2D *mat,
    StridedMemRefType_f32_1D *vec,
    StridedMemRefType_i32_1D *perm
);


static StridedMemRefType_f32_2D make_memref_2d_f32(long long rows, long long cols) {
    StridedMemRefType_f32_2D mr;
    size_t bytes = (size_t)(rows * cols) * sizeof(float);
    float *data = (float*)aligned_alloc(64, (bytes + 63) & ~63);
    mr.allocated = mr.aligned = data;
    mr.offset = 0; mr.sizes[0] = rows; mr.sizes[1] = cols;
    mr.strides[0] = cols; mr.strides[1] = 1;
    return mr;
}

static StridedMemRefType_f32_1D make_memref_1d_f32(long long size) {
    StridedMemRefType_f32_1D mr;
    size_t bytes = size * sizeof(float);
    float *data = (float*)aligned_alloc(64, (bytes + 63) & ~63);
    mr.allocated = mr.aligned = data;
    mr.offset = 0; mr.sizes[0] = size; mr.strides[0] = 1;
    return mr;
}

static StridedMemRefType_i32_1D make_memref_1d_i32(long long size) {
    StridedMemRefType_i32_1D mr;
    size_t bytes = size * sizeof(int);
    int *data = (int*)aligned_alloc(64, (bytes + 63) & ~63);
    mr.allocated = mr.aligned = data;
    mr.offset = 0; mr.sizes[0] = size; mr.strides[0] = 1;
    return mr;
}

static void free_memref_2d_f32(StridedMemRefType_f32_2D *mr) { free(mr->allocated); }
static void free_memref_1d_f32(StridedMemRefType_f32_1D *mr) { free(mr->allocated); }
static void free_memref_1d_i32(StridedMemRefType_i32_1D *mr) { free(mr->allocated); }

static void random_fill_lu_mat(StridedMemRefType_f32_2D *mr) {
    long long dim = mr->sizes[0];
    for (long long i = 0; i < dim; i++) {
        float row_sum = 0.0f;

        // 1. Prima genero tutti i numeri casuali e sommo i valori assoluti
        for (long long j = 0; j < dim; j++) {
            float val = (float)((rand() % 200) - 100) / 10.0f;
            mr->aligned[i * dim + j] = val;
            if (i != j) {
                row_sum += fabsf(val);
            }
        }

        // 2. Metto sulla diagonale un valore SICURAMENTE più grande della somma della riga
        // Questo garantisce una matrice rigorosamente a diagonale dominante (stabilissima!)
        mr->aligned[i * dim + i] = row_sum + 1.0f;
    }
}

static void random_fill_vec(StridedMemRefType_f32_1D *mr) {
    for (long long i = 0; i < mr->sizes[0]; i++) {
        mr->aligned[i] = (float)((rand() % 200) - 100) / 10.0f;
    }
}

static void fill_identity_perm(StridedMemRefType_i32_1D *mr) {
    for (long long i = 0; i < mr->sizes[0]; i++) {
        mr->aligned[i] = (int)i; // Permutazione identità per semplicità
    }
}

//goldenstandard
static int linalg_lu_solve_pulp_open_fc(const float *mat, const float *vec, const int *perm, float *result, const int dim_M, const int dim_N)
{
    int perm_idx;
    float sum;

    float *y = (float*)malloc(dim_M * sizeof(float)); 

    for (int m = 0; m < dim_M; m++) {
        sum = 0;
        perm_idx = perm[m];

        for (int k = 0; k < m; k++)
            sum += mat[m * dim_N + k] * y[k];
        y[m] = vec[perm_idx] - sum;
    }

    for (int m = (dim_M - 1); m >= 0; m--) {
        sum = 0;

        for (int k = (m + 1); k < dim_N; k++)
            sum += mat[m * dim_N + k] * result[k];
        result[m] = (y[m] - sum) / mat[m * dim_N + m];
    }

    free(y); 
    return 0;
}


static double error_check(const float *MLIR_ptr, const float *C_ptr, long long elements) {
    double error = 0.0;
    for (long long i = 0; i < elements; i++) {
        double diff = (double)MLIR_ptr[i] - (double)C_ptr[i];
        error += diff * diff;
    }
    return error / elements;
}


int main(int argc, char **argv) {
    long long dim_M = 8;
    if (argc == 2) dim_M = atoll(argv[1]);

    // Input originari
    StridedMemRefType_f32_2D mat = make_memref_2d_f32(dim_M, dim_M);
    StridedMemRefType_f32_1D vec = make_memref_1d_f32(dim_M);
    StridedMemRefType_i32_1D perm = make_memref_1d_i32(dim_M);
    
    // Buffer per i calcoli in C
    float *c_result = (float*)malloc(dim_M * sizeof(float));

    printf("Test LU Solve (Dim=%lld)...\n", dim_M);

    for (int iter = 0; iter < 5; iter++) {
        srand(42 + iter);
        
        random_fill_lu_mat(&mat);
        random_fill_vec(&vec);
        fill_identity_perm(&perm);

        StridedMemRefType_f32_1D mlir_out;

        _mlir_ciface_linalg_lu_solve(&mlir_out, &mat, &vec, &perm);

        linalg_lu_solve_pulp_open_fc(mat.aligned, vec.aligned, perm.aligned, c_result, dim_M, dim_M);

        double err = error_check(mlir_out.aligned, c_result, dim_M);
        if(err < 1e-3) {
            printf("Esito: SUCCESSO, Errore = %e\n", err);
        } else {
            printf("Esito: FALLITO, Errore = %e\n", err);
        }
        
        free(mlir_out.allocated);
    }

    free_memref_2d_f32(&mat);
    free_memref_1d_f32(&vec);
    free_memref_1d_i32(&perm);
    free(c_result); 
    
    return 0;
}
