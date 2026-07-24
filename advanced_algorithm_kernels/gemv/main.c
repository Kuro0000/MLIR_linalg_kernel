#include <stdio.h>
#include <stdlib.h>
#include <math.h>

typedef struct {
  float *allocated;
  float *aligned;
  long long offset;
  long long sizes[2];
  long long strides[2];
} StridedMemRefType_f32_2D;

typedef struct {
  float *allocated;
  float *aligned;
  long long offset;
  long long sizes[1];
  long long strides[1];
} StridedMemRefType_f32_1D;


extern void _mlir_ciface_linalg_gemv(
    StridedMemRefType_f32_1D *out,
    StridedMemRefType_f32_2D *mat,
    StridedMemRefType_f32_1D *vec_x,
    StridedMemRefType_f32_1D *vec_y,
    float alpha,
    float beta
);


static StridedMemRefType_f32_2D make_memref_2d(long long M, long long N) {
  StridedMemRefType_f32_2D mr;
  size_t bytes = (size_t)(M * N) * sizeof(float);
  float *data = (float*)aligned_alloc(64, bytes);
  mr.allocated = data;
  mr.aligned   = data;
  mr.offset    = 0;
  mr.sizes[0]   = M;
  mr.sizes[1]   = N;
  mr.strides[0] = N;
  mr.strides[1] = 1;
  return mr;
}

static StridedMemRefType_f32_1D make_memref_1d(long long N) {
  StridedMemRefType_f32_1D mr;
  size_t bytes = (size_t)N * sizeof(float);
  float *data = (float*)aligned_alloc(64, bytes);
  mr.allocated = data;
  mr.aligned   = data;
  mr.offset    = 0;
  mr.sizes[0]   = N;
  mr.strides[0] = 1;
  return mr;
}

static void free_memref_2d(StridedMemRefType_f32_2D *mr) {
  if (mr->allocated) free(mr->allocated);
  mr->allocated = mr->aligned = NULL;
}

static void free_memref_1d(StridedMemRefType_f32_1D *mr) {
  if (mr->allocated) free(mr->allocated);
  mr->allocated = mr->aligned = NULL;
}

static void random_fill_2d(StridedMemRefType_f32_2D *mr, unsigned seed) {
  srand(seed);
  long long total_elements = mr->sizes[0] * mr->sizes[1];
  for (long long i = 0; i < total_elements; i++) {
    mr->aligned[mr->offset + i] = (float)((rand() % 2001) - 1000) / 100.0f;
  }
}

static void random_fill_1d(StridedMemRefType_f32_1D *mr, unsigned seed) {
  srand(seed);
  long long total_elements = mr->sizes[0];
  for (long long i = 0; i < total_elements; i++) {
    mr->aligned[mr->offset + i] = (float)((rand() % 2001) - 1000) / 100.0f;
  }
}
//golden standard
const float ZERO_f = 0.0f;
const float ONE_f = 1.0f;

static inline void vector_set_all(float *dst, float val, int len) {
    for(int i = 0; i < len; i++) dst[i] = val;
}

static inline void vector_memcpy(const float *src, float *dst, int len) {
    for(int i = 0; i < len; i++) dst[i] = src[i];
}

static inline void vector_scale(const float *src, float scale, float *dst, int len) {
    for(int i = 0; i < len; i++) dst[i] = src[i] * scale;
}

static int linalg_gemv_pulp_open_fc(const float *mat, const float *vec_x, const float *vec_y, const float alpha, const float beta, float *dst, const int dim_M, const int dim_N)
{
    float sum;

    if (alpha == ZERO_f) {
        if (beta == ZERO_f)
            vector_set_all(dst, ZERO_f, dim_M);
        else if (beta == ONE_f)
            vector_memcpy(vec_y, dst, dim_M);
        else
            vector_scale(vec_y, beta, dst, dim_M);
        return 0;
    }

    for (int m = 0; m < dim_M; m++) {
        sum = 0;
        for (int n = 0; n < dim_N; n++)
            sum += mat[m * dim_N + n] * vec_x[n];
        dst[m] = alpha * sum + beta * vec_y[m];
    }

    return 0;
}

static double error_check_1d(const StridedMemRefType_f32_1D *X, const float *Ref) {
  long long N = X->sizes[0];
  double s = 0.0;
  for (long long i = 0; i < N; i++) {
      double dx = (double)X->aligned[X->offset + i] - (double)Ref[i];
      s += dx * dx;
  }
  return s;
}

static void print_vector_head(const char *name, const float *vec, long long N, long long limit) {
    long long k = (limit < N) ? limit : N;
    printf("%s: [", name);
    for (long long i = 0; i < k; i++) {
        printf("%7.2f ", vec[i]);
    }
    if (k < N) printf("...");
    printf("]\n");
}

//main
int main(int argc, char **argv) {
    long long M = 8, N = 8;
    if (argc >= 3) {
        M = atoll(argv[1]);
        N = atoll(argv[2]);
    }

    StridedMemRefType_f32_2D Mat   = make_memref_2d(M, N);
    StridedMemRefType_f32_1D VecX  = make_memref_1d(N);
    StridedMemRefType_f32_1D VecY  = make_memref_1d(M);
    
    float *ref_dst = (float*)malloc(M * sizeof(float));


    float test_alphas[] = {1.0f, 1.0f, 0.5f, 0.0f, 0.0f};
    float test_betas[]  = {1.0f, 0.0f, 2.0f, 1.0f, 0.0f};
    int num_tests = 5;

    for (int i = 0; i < num_tests; i++) {
        float alpha = test_alphas[i];
        float beta  = test_betas[i];

        random_fill_2d(&Mat,  13 + i);
        random_fill_1d(&VecX, 42 + i);
        random_fill_1d(&VecY, 99 + i);

        linalg_gemv_pulp_open_fc(Mat.aligned, VecX.aligned, VecY.aligned, alpha, beta, ref_dst, (int)M, (int)N);

        StridedMemRefType_f32_1D mlir_dst = {0}; 
        _mlir_ciface_linalg_gemv(&mlir_dst, &Mat, &VecX, &VecY, alpha, beta);

        double err = error_check_1d(&mlir_dst, ref_dst);

        if (err < 1e-4) {
            printf("  Esito: SUCCESSO, Errore = %e\n", err);
        } else {
            printf("  Esito: FALLITO, Errore = %e\n", err);
            print_vector_head("  MLIR Output", mlir_dst.aligned, M, 8);
            print_vector_head("  C Reference", ref_dst, M, 8);
        }

        free_memref_1d(&mlir_dst);
    }

    free_memref_2d(&Mat);
    free_memref_1d(&VecX);
    free_memref_1d(&VecY);
    free(ref_dst);

    return 0;
}