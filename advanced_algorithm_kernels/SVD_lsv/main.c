#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

static float EPSILON = 1e-12;

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

static StridedMemRefType_f32_1D make_memref_1d(long long N) {
  StridedMemRefType_f32_1D mr;
  size_t nbytes = (size_t)N * sizeof(float);
  float *data  = (float *)aligned_alloc(64, nbytes);
  memset(data, 0, nbytes);
  mr.allocated  = data;
  mr.aligned    = data;
  mr.offset     = 0;
  mr.sizes[0]   = N;
  mr.strides[0] = 1;
  return mr;
}

static void free_memref_2d(StridedMemRefType_f32_2D *mr) {
  free(mr->allocated);
  mr->allocated = mr->aligned = NULL;
}

static void free_memref_1d(StridedMemRefType_f32_1D *mr) {
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
static inline void set1D(StridedMemRefType_f32_1D *m, long long i, float v) {
  m->aligned[m->offset + i * m->strides[0]] = v;
}


extern void _mlir_ciface_linalg_svd_lsv(
    StridedMemRefType_f32_2D *dst,
    StridedMemRefType_f32_2D *src,
    StridedMemRefType_f32_2D *mat_V,
    StridedMemRefType_f32_1D *vec_S);

static int linalg_svd_lsv_pulp_open_fc(const float *src, float *mat_V, float *vec_S, float *dst, const int dim_M, const int dim_N)
{
    float sum;

    for (int k = 0; k < dim_N; k++) {
        /* Optimization */
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

static void print_matrix(const char *name, const StridedMemRefType_f32_2D *mr) {
  long long R = mr->sizes[0], C = mr->sizes[1];
  printf("\n%s (%lld x %lld):\n", name, R, C);
  for (long long r = 0; r < R; r++) {
    for (long long c = 0; c < C; c++)
      printf("%9.4f ", get2D(mr, r, c)  );
    printf("\n");
  }
}


int main(int argc, char **argv) {
  long long M = 4;
  if (argc >= 2) M = atoll(argv[1]);

  for (int trial = 1; trial <= 6; trial++) {
    printf("\nTEST %d\n", trial);

    srand((unsigned)(trial * 13));

    // Matrice sorgente (input)
    StridedMemRefType_f32_2D src = make_memref_2d(M, M);
    for (long long i = 0; i < M; i++)
      for (long long j = 0; j < M; j++)
        set2D(&src, i, j, (float)((rand() % 2001) - 1000) / 100.0f);

    // Vettori singolari destri V (input)
    StridedMemRefType_f32_2D mat_V = make_memref_2d(M, M);
    for (long long i = 0; i < M; i++)
      for (long long j = 0; j < M; j++)
        set2D(&mat_V, i, j, (float)((rand() % 2001) - 1000) / 100.0f);

    // Valori singolari S > 0 (input)
    StridedMemRefType_f32_1D vec_S = make_memref_1d(M);
    for (long long i = 0; i < M; i++)
      set1D(&vec_S, i, (float)(rand() % 100 + 1) / 10.0f);

    // Output MLIR
    StridedMemRefType_f32_2D mlir_dst = make_memref_2d(M, M);
    _mlir_ciface_linalg_svd_lsv(&mlir_dst, &src, &mat_V, &vec_S);

    // Output riferimento (golden standard)
    StridedMemRefType_f32_2D ref_dst = make_memref_2d(M, M);
    linalg_svd_lsv_pulp_open_fc(
      src.aligned,
      mat_V.aligned,
      vec_S.aligned,
      ref_dst.aligned,
      (int)M,
      (int)M);

    // Calcolo errore elemento per elemento
    double err = 0.0;
    for (long long i = 0; i < M; i++)
      for (long long j = 0; j < M; j++) {
        double d = (double)get2D(&mlir_dst, i, j) - (double)get2D(&ref_dst, i, j);
        err += d * d;
      }
    printf("  Errore: %.6e\n", err);
    if (err < 1e-3)
      printf("  Esito: SUPERATO, Errore = %e\n", err);
    else
      printf("  Esito: FALLITO, Errore = %e\n", err);

    print_matrix("dst MLIR", &mlir_dst);
    print_matrix("dst Ref",  &ref_dst);

    free_memref_2d(&src);
    free_memref_2d(&mat_V);
    free_memref_1d(&vec_S);
    free_memref_2d(&mlir_dst);
    free_memref_2d(&ref_dst);
  }
  return 0;
}

