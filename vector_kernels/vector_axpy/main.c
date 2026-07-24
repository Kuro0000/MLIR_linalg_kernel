#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>

typedef struct {
  float *allocated;
  float *aligned;
  long long offset;
  long long sizes[1];
  long long strides[1];
} StridedMemRefType_f32_1D;

extern void _mlir_ciface_vector_axpy(StridedMemRefType_f32_1D *dst,
                                    StridedMemRefType_f32_1D *src_a,
                                    StridedMemRefType_f32_1D *src_b,
                                    float alpha);

static StridedMemRefType_f32_1D make_memref_1d(long long N) {
  StridedMemRefType_f32_1D mr;
  size_t bytes = (size_t)N * sizeof(float);
  float *data = (float*)aligned_alloc(64, bytes);
  mr.allocated = mr.aligned = data;
  mr.offset    = 0;
  mr.sizes[0]  = N;
  mr.strides[0]= 1;
  return mr;
}

static void free_memref(StridedMemRefType_f32_1D *mr) {
  if (mr && mr->allocated) {
    free(mr->allocated);
    mr->allocated = mr->aligned = NULL;
  }
}

// Riempie il vettore con valori casuali
static void random_fill(StridedMemRefType_f32_1D *mr, unsigned seed) {
  srand(seed);
  long long N = mr->sizes[0];
  for (long long i = 0; i < N; i++) {
    mr->aligned[mr->offset + i * mr->strides[0]] =
        (float)((rand() % 2001) - 1000) / 100.0f;
  }
}
//golden standard
static void ref_vector_axpy(StridedMemRefType_f32_1D *dst,
                           StridedMemRefType_f32_1D *src_a,
                           StridedMemRefType_f32_1D *src_b,
                           float alpha) {
    long long len = dst->sizes[0];
    for (long long i = 0; i < len; i++) {
        float v = src_a->aligned[src_a->offset + i * src_a->strides[0]] * alpha
                  + src_b->aligned[src_b->offset + i * src_b->strides[0]];
        dst->aligned[dst->offset + i * dst->strides[0]] = v;
    }
}

static void print_vector(StridedMemRefType_f32_1D *vec, const char *name, int max_print) {
    printf("%s = [", name);
    long long N = vec->sizes[0];
    int print_count = (N < max_print) ? N : max_print;
    
    for (long long i = 0; i < print_count; i++) {
        printf("%s%.3f", i > 0 ? ", " : "", 
               vec->aligned[vec->offset + i * vec->strides[0]]);
    }
    if (N > max_print) printf(", ...");
    printf("]\n");
}

int main(int argc, char **argv) {
  long long N = 64; 
  if (argc == 2) { N = atoll(argv[1]); }
  srand(time(NULL));

  printf("=== INIZIO TEST VECTOR AXPY TENSOR (Dimensione: %lld) ===\n\n", N);
  
  StridedMemRefType_f32_1D dst_mlir; 
  
  StridedMemRefType_f32_1D dst_ref  = make_memref_1d(N);
  StridedMemRefType_f32_1D src_a    = make_memref_1d(N);
  StridedMemRefType_f32_1D src_b    = make_memref_1d(N);
  float alpha = 2.5f;

  for(int test = 1; test <= 6; test++) {
    random_fill(&src_a, 13 + test);
    random_fill(&src_b, 17 + test);

    printf("Test %d:\n", test);
    print_vector(&src_a, "  src_a           ", 8);
    print_vector(&src_b, "  src_b           ", 8);
    printf("  alpha = %.3f\n", alpha);

    _mlir_ciface_vector_axpy(&dst_mlir, &src_a, &src_b, alpha);
    
    ref_vector_axpy(&dst_ref, &src_a, &src_b, alpha);

    double max_err = 0.0;
    for (long long i = 0; i < N; i++) {
      double err = fabs((double)dst_mlir.aligned[i] - (double)dst_ref.aligned[i]);
      if (err > max_err) max_err = err;
    }

    print_vector(&dst_mlir, "  Risultato MLIR  ", 8);
    print_vector(&dst_ref, "  Risultato REF   ", 8);
    printf("  Errore massimo: %.6e\n", max_err);
    printf("  Esito: %s\n\n", (max_err < 1e-5) ? "SUPERATO" : "FALLITO");
    
    if (test < 6) {
        free(dst_mlir.allocated);
    }
  }

  free(dst_mlir.allocated);
  free_memref(&dst_ref);
  free_memref(&src_a);
  free_memref(&src_b);

  return 0;
}