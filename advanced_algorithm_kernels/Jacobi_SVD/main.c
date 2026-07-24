#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

typedef struct {
  float    *allocated;
  float    *aligned;
  long long offset;
  long long sizes[2];
  long long strides[2];
} StridedMemRefType_f32_2D;

typedef struct {
  float    *allocated;
  float    *aligned;
  long long offset;
  long long sizes[1];
  long long strides[1];
} StridedMemRefType_f32_1D;

typedef struct {
  StridedMemRefType_f32_2D mat;   // diagonalised mat
  StridedMemRefType_f32_2D V;     // right singular vectors
  StridedMemRefType_f32_1D S;     // singular values
} SVDResult;


static StridedMemRefType_f32_2D make_memref_2d(long long M, long long N) {
  StridedMemRefType_f32_2D mr;
  size_t nbytes = (size_t)M * (size_t)N * sizeof(float);
  float *data  = (float *)aligned_alloc(64, nbytes);
  memset(data, 0, nbytes);
  mr.allocated  = data;
  mr.aligned    = data;
  mr.offset     = 0;
  mr.sizes[0]   = M;  mr.sizes[1]   = N;
  mr.strides[0] = N;  mr.strides[1] = 1;
  return mr;
}

static void free_memref_2d(StridedMemRefType_f32_2D *mr) {
  free(mr->allocated);
  mr->allocated = mr->aligned = NULL;
}

static inline float get2D(const StridedMemRefType_f32_2D *m, long long r, long long c) {
  return m->aligned[m->offset + r * m->strides[0] + c * m->strides[1]];
}
static inline void set2D(StridedMemRefType_f32_2D *m, long long r, long long c, float v) {
  m->aligned[m->offset + r * m->strides[0] + c * m->strides[1]] = v;
}
static inline float get1D(const StridedMemRefType_f32_1D *m, long long i) {
  return m->aligned[m->offset + i * m->strides[0]];
}

// Build a random symmetric PSD matrix A = R^T * R
static StridedMemRefType_f32_2D make_random_psd(long long M, unsigned seed) {
  StridedMemRefType_f32_2D R = make_memref_2d(M, M);
  srand(seed);
  for (long long i = 0; i < M; i++)
    for (long long j = 0; j < M; j++)
      set2D(&R, i, j, (float)((rand() % 2001) - 1000) / 100.0f);
  StridedMemRefType_f32_2D A = make_memref_2d(M, M);
  for (long long i = 0; i < M; i++)
    for (long long j = 0; j < M; j++) {
      float s = 0.0f;
      for (long long k = 0; k < M; k++)
        s += get2D(&R, k, i) * get2D(&R, k, j);
      set2D(&A, i, j, s);
    }
  free_memref_2d(&R);
  return A;
}

// MLIR function
extern void _mlir_ciface_linalg_svd_jacobi(SVDResult *result, StridedMemRefType_f32_2D *mat_in);

// Reference Jacobi SVD
static void jacobi_svd_ref(float *mat, float *mat_V, float *vec_S, int dim_M) {
  const float EPSILON  = 1e-12f;
  const int   MAX_ITER = 200;
  memset(mat_V, 0, (size_t)dim_M * dim_M * sizeof(float));
  for (int i = 0; i < dim_M; i++) mat_V[i*dim_M+i] = 1.0f;
  for (int iter = 0; iter < MAX_ITER; iter++) {
    float max_off = 0.0f;
    for (int i = 0; i < dim_M - 1; i++) {
      for (int j = i + 1; j < dim_M; j++) {
        float mij = mat[i*dim_M+j];
        if (fabsf(mij) < EPSILON) continue;
        float tau = (mat[j*dim_M+j] - mat[i*dim_M+i]) / (2.0f * mij);
        float t   = (tau >= 0.0f) ? 1.0f/(tau+sqrtf(1.0f+tau*tau))
                                   : 1.0f/(tau-sqrtf(1.0f+tau*tau));
        float c = 1.0f/sqrtf(1.0f+t*t), s = t*c;
        for (int m = 0; m < dim_M; m++) {
          float im=mat[i*dim_M+m], jm=mat[j*dim_M+m];
          mat[i*dim_M+m]=c*im-s*jm; mat[j*dim_M+m]=s*im+c*jm;
        }
        for (int n = 0; n < dim_M; n++) {
          float ni=mat[n*dim_M+i], nj=mat[n*dim_M+j];
          mat[n*dim_M+i]=c*ni-s*nj; mat[n*dim_M+j]=s*ni+c*nj;
        }
        for (int n = 0; n < dim_M; n++) {
          float vi=mat_V[n*dim_M+i], vj=mat_V[n*dim_M+j];
          mat_V[n*dim_M+i]=c*vi-s*vj; mat_V[n*dim_M+j]=s*vi+c*vj;
        }
        float post = fabsf(mat[i*dim_M+j]);
        if (post > max_off) max_off = post;
      }
    }
    if (max_off < EPSILON) break;
  }
  for (int i = 0; i < dim_M; i++)
    vec_S[i] = mat[i*dim_M+i] > 0.0f ? sqrtf(mat[i*dim_M+i]) : 0.0f;
}

static void print_vector_mlir(const char *name, const StridedMemRefType_f32_1D *v) {
  long long M = v->sizes[0];
  printf("\n%s (%lld):\n", name, M);
  for (long long i = 0; i < M; i++)
    printf("%9.4f ", get1D(v, i));
  printf("\n");
}

static void print_vector_ref(const char *name, const float *v, int M) {
  printf("\n%s (%d):\n", name, M);
  for (int i = 0; i < M; i++)
     printf("%9.4f ", v[i]);
  printf("\n");
}

static void print_matrix_mlir(const char *name, const StridedMemRefType_f32_2D *mr, long long k) {
  long long R=mr->sizes[0], C=mr->sizes[1];
  printf("\n%s (%lld x %lld):\n", name, R, C);
  for (long long r = 0; r < R; r++) {
    for (long long c = 0; c < C; c++)
      printf("%9.4f ", get2D(mr, r, c));
    printf("\n");
  }
}

static void print_matrix_ref(const char *name, const float *mr, int M, int N) {
  printf("\n%s (%dx%d):\n", name, M, N);
  for (int i = 0; i < M; i++) {
    for (int j = 0; j < N; j++)
      printf("%9.4f ", mr[i*N+j]);
    printf("\n");
  }
}


int main(int argc, char **argv) {
  long long M = 4;
  if (argc >= 2) M = atoll(argv[1]);

  for (int trial = 1; trial <= 6; trial++) {
    printf("\nTEST %d\n", trial);

    // MLIR execution
    StridedMemRefType_f32_2D A     = make_random_psd(M, (unsigned)(trial * 13));
    StridedMemRefType_f32_2D A_ref = make_memref_2d(M, M);
    memcpy(A_ref.aligned, A.aligned, (size_t)M * M * sizeof(float));
    SVDResult result = {0};
    _mlir_ciface_linalg_svd_jacobi(&result, &A);

    // Reference execution
    float *ref_mat = malloc((size_t)M * M * sizeof(float));
    float *ref_V   = malloc((size_t)M * M * sizeof(float));
    float *ref_S   = malloc((size_t)M     * sizeof(float));
    memcpy(ref_mat, A_ref.aligned, (size_t)M * M * sizeof(float));
    jacobi_svd_ref(ref_mat, ref_V, ref_S, (int)M);

    // Error check
    double err = 0.0;
    for (long long i = 0; i < M; i++) {
      double d = (double)get1D(&result.S, i) - (double)ref_S[i];
      err += d * d;
    }
    printf("Error: %.6e\n", err);

    // Print values
    print_vector_mlir("Singular values (MLIR)", &result.S);
    print_vector_ref ("Singular values C (ref)", ref_S, (int)M);
    print_matrix_mlir("V (MLIR)", &result.V, 4);
    print_matrix_ref("V (ref)", ref_V, (int)M, (int)M);

    free(ref_mat); free(ref_V); free(ref_S);
    free_memref_2d(&A);
    free_memref_2d(&A_ref);
  }
  return 0;
}

