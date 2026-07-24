#include <stdio.h>
#include <stdlib.h>
#include <math.h>

static float ONE_f = 1.0f;

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
extern void _mlir_ciface_linalg_cholesky_decomp(
    StridedMemRefType_f32_2D *out_dst,
    StridedMemRefType_f32_2D *src
);



static void free_memref_2d(StridedMemRefType_f32_2D *mr) { 
    free(mr->allocated); 
}

static void zero_fill_2d(StridedMemRefType_f32_2D *mr) {
    long long total = mr->sizes[0] * mr->sizes[1];
    for (long long i = 0; i < total; i++) {
        mr->aligned[i] = 0.0f;
    }
}

// Generatore di Matrice Simmetrica Definita Positiva (A * A^T)
static void spd_fill_2d(StridedMemRefType_f32_2D *mr) {
    long long dim = mr->sizes[0];
    float *temp_A = (float*)malloc(dim * dim * sizeof(float));
    
    for (long long i = 0; i < dim * dim; i++) {
        // Range [-10.0, 10.0] con passo 0.01
        temp_A[i] = ((float)(rand() % 2001) - 1000.0f) / 100.0f;
    }
    
    for (long long i = 0; i < dim; i++) {
        for (long long j = 0; j < dim; j++) {
            float sum = 0.0f;
            for (long long k = 0; k < dim; k++) {
                sum += temp_A[i * dim + k] * temp_A[j * dim + k];
            }
            mr->aligned[i * dim + j] = sum;
            
            if (i == j) {
                mr->aligned[i * dim + j] += 5.0f; 
            }
        }
    }
    free(temp_A);
}

// Golden Standard 
static int linalg_cholesky_decomp_pulp_open_fc(const float *src, float *dst,  const int dim)
{
    for (int m = 0; m < dim; m++) {
        float sum = 0;
        for (int n = 0; n < m; n++)
            sum += dst[m * dim + n] * dst[m * dim + n];
        dst[m * dim + m] = sqrtf(src[m * dim + m] - sum);

        for (int n = (m + 1); n < dim; n++) {
            float sum = 0;
            for (int k = 0; k < m; k++) {
                sum += dst[n * dim + k] * dst[m * dim + k];
            }
            dst[n * dim + m] = (ONE_f / dst[m * dim + m]) * (src[n * dim + m] - sum);
        }
    }

    return 0;
}


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

//main

int main(int argc, char **argv) {
    long long dim_M = 64;
    if (argc == 2) dim_M = atoll(argv[1]);

    StridedMemRefType_f32_2D src = make_memref_2d(dim_M, dim_M);
    StridedMemRefType_f32_2D dst_c = make_memref_2d(dim_M, dim_M);

    printf("Test Cholesky Decomposition (Dim=%lld)...\n", dim_M);

    for (int iter = 0; iter < 5; iter++) {
        srand(42 + iter);
        
        spd_fill_2d(&src);     
        zero_fill_2d(&dst_c);   
        
        StridedMemRefType_f32_2D dst_mlir;

        _mlir_ciface_linalg_cholesky_decomp(&dst_mlir, &src);

        linalg_cholesky_decomp_pulp_open_fc(src.aligned, dst_c.aligned, dim_M);

        double err = error_check(&dst_mlir, &dst_c);

        if(err < 1e-3) {
            printf("Esito: SUCCESSO, Errore = %e\n", err);
        } else {
            printf("Esito: FALLITO, Errore = %e\n", err);
        }
        free(dst_mlir.allocated);
    }

    free_memref_2d(&src);
    free_memref_2d(&dst_c);
    
    return 0;
}