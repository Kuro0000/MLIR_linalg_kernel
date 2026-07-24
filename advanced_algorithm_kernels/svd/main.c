#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#define EPSILON 1e-12f


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
  StridedMemRefType_f32_2D dst;   // Matrice ricostruita / processata
  StridedMemRefType_f32_2D mat_V; // Vettori singolari destri
  StridedMemRefType_f32_1D vec_S; // Valori singolari
} SVD_Full_Result;

extern void _mlir_ciface_linalg_svd_pulp_open_fc(SVD_Full_Result *out, StridedMemRefType_f32_2D *src);


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
  if (mr->allocated) free(mr->allocated);
  mr->allocated = mr->aligned = NULL;
}

static void free_memref_1d(StridedMemRefType_f32_1D *mr) {
  if (mr->allocated) free(mr->allocated);
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

static void print_matrix(const char *name, const float *mr, int M, int N) {
  printf("\n%s (%dx%d):\n", name, M, N);
  for (int i = 0; i < M; i++) {
    for (int j = 0; j < N; j++)
      printf("%9.4f ", mr[i*N+j]);
    printf("\n");
  }
}



static int matrix_mul_trans_A_pulp_open_fc(const float *src_a, const float *src_b, float *dst, const int dim_M, const int dim_N, const int dim_P) {
    float sum;
    for (int n = 0; n < dim_N; n++) {
        for (int p = 0; p < dim_P; p++) {
            sum = 0;
            for (int m = 0; m < dim_M; m++)
                sum += src_a[m * dim_N + n] * src_b[m * dim_P + p];
            dst[n * dim_P + p] = sum;
        }
    }
    return 0;
}

static void jacobi_svd_ref(float *mat, float *mat_V, float *vec_S, int dim_M) {
    const int MAX_ITER = 200;
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

static int linalg_svd_lsv_pulp_open_fc(const float *src, float *mat_V, float *vec_S, float *dst, const int dim_M, const int dim_N) {
    float sum;
    for (int k = 0; k < dim_N; k++) {
        if (vec_S[k] < EPSILON) {
            for (int m = 0; m < dim_M; m++)
                dst[m * dim_N + k] = 0;
            continue;
        }
        for (int m = 0; m < dim_M; m++) {
            sum = 0;
            for (int n = 0; n < dim_N; n++)
                sum += src[m * dim_N + n] * mat_V[n * dim_N + k];
            dst[m * dim_N + k] = sum / vec_S[k];
        }
    }
    return 0;
}

//golden standard
static int linalg_svd_pulp_open_fc_ref(const float *src, float *dst, float *mat_V, float *vec_S, const int dim_M, const int dim_N) {
    float *tmp = (float*)malloc(dim_N * dim_N * sizeof(float));
    
    matrix_mul_trans_A_pulp_open_fc(src, src, tmp, dim_M, dim_N, dim_N);
    jacobi_svd_ref(tmp, mat_V, vec_S, dim_N);
    linalg_svd_lsv_pulp_open_fc(src, mat_V, vec_S, dst, dim_M, dim_N);
    
    free(tmp);
    return 0;
}



//main
int main(int argc, char **argv) {
  long long M = 4;
  long long N = 4;
  if (argc >= 3) {
      M = atoll(argv[1]);
      N = atoll(argv[2]);
  }

  for (int trial = 1; trial <= 6; trial++) {


    srand((unsigned)(trial * 13));

    StridedMemRefType_f32_2D src = make_memref_2d(M, N);
    for (long long i = 0; i < M; i++) {
      for (long long j = 0; j < N; j++) {
        set2D(&src, i, j, (float)((rand() % 2001) - 1000) / 100.0f);
      }
    }

    float *ref_dst   = (float*)malloc(M * N * sizeof(float));
    float *ref_mat_V = (float*)malloc(N * N * sizeof(float));
    float *ref_vec_S = (float*)malloc(N     * sizeof(float));

    linalg_svd_pulp_open_fc_ref(src.aligned, ref_dst, ref_mat_V, ref_vec_S, (int)M, (int)N);


    SVD_Full_Result mlir_res = {0}; 
    _mlir_ciface_linalg_svd_pulp_open_fc(&mlir_res, &src);

    double err_dst = 0.0, err_V = 0.0, err_S = 0.0;
    
    for (long long i = 0; i < N; i++) {
        double ds = (double)get1D(&mlir_res.vec_S, i) - (double)ref_vec_S[i];
        err_S += ds * ds;
    }
    
    for (long long i = 0; i < N; i++) {
      for (long long j = 0; j < N; j++) {
        double dv = (double)get2D(&mlir_res.mat_V, i, j) - (double)ref_mat_V[i*N+j];
        err_V += dv * dv;
      }
    }

    for (long long i = 0; i < M; i++) {
      for (long long j = 0; j < N; j++) {
        double dd = (double)get2D(&mlir_res.dst, i, j) - (double)ref_dst[i*N+j];
        err_dst += dd * dd;
      }
    }

    printf("Error scalare : %.6e\n", err_S);
    printf("Error vettore: %.6e\n", err_V);
    printf("Error matrice: %.6e\n", err_dst);

    if (err_dst < 1e-3 && err_V < 1e-3 && err_S < 1e-3)
      printf("\nESITO: SUPERATO, Errore = %e\n", err_dst);
    else
      printf("\nESITO: FALLITO, Errore = %e\n", err_dst);

    free_memref_2d(&mlir_res.dst);
    free_memref_2d(&mlir_res.mat_V);
    free_memref_1d(&mlir_res.vec_S);
    
    // Deallocazione buffers
    free_memref_2d(&src);
    free(ref_dst);
    free(ref_mat_V);
    free(ref_vec_S);
  }
  
  return 0;
}