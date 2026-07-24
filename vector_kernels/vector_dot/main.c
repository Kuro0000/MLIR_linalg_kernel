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

extern float _mlir_ciface_vector_dot(StridedMemRefType_f32_1D *A, 
                                     StridedMemRefType_f32_1D *B);

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

// Libera la memoria
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
// Golden Model 
static void ref_vector_dot(float *out, StridedMemRefType_f32_1D *A, StridedMemRefType_f32_1D *B) {
    long long len = A->sizes[0];
    *out = 0.0f;
    for (long long idx = 0; idx < len; idx++) {
        *out += A->aligned[A->offset + idx * A->strides[0]] * B->aligned[B->offset + idx * B->strides[0]];
    }
}

// Stampa il vettore 
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

  StridedMemRefType_f32_1D vecA = make_memref_1d(N);
  StridedMemRefType_f32_1D vecB = make_memref_1d(N);

  printf("=== INIZIO TEST VECTOR DOT (Dimensione: %lld) ===\n\n", N);

  for(int test = 1; test <= 6; test++) {
      random_fill(&vecA, 13 + test);
      random_fill(&vecB, 17 + test); 
      
      printf("Test %d:\n", test);
      print_vector(&vecA, "  Vettore A       ", 8);
      print_vector(&vecB, "  Vettore B       ", 8);
      
      // Esecuzione MLIR - Restituisce direttamente il float
      float result_mlir = _mlir_ciface_vector_dot(&vecA, &vecB);

      // Esecuzione C standard
      float result_ref = 0.0f;
      ref_vector_dot(&result_ref, &vecA, &vecB);

      // Calcolo dell'errore (solo sullo scalare risultante)
      double err = fabs((double)result_mlir - (double)result_ref);
      
      // Output dei risultati visibili come richiesto
      printf("  Risultato MLIR  : %.6f\n", result_mlir);
      printf("  Risultato C Ref : %.6f\n", result_ref);
      printf("  Errore assoluto : %.6e\n", err);
      
      if (err < 1e-3) {
          printf("  Esito: SUPERATO, Errore = %e\n\n", err);
      } else {
          printf("  Esito: FALLITO, Errore = %e\n\n", err);
      }
  }

  free_memref(&vecA);
  free_memref(&vecB);
  
  return 0;
}