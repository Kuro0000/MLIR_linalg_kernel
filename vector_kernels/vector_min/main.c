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

extern float _mlir_ciface_vector_min(StridedMemRefType_f32_1D *A);

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

// Riempie il vettore con valori casuali [-10, 10]
static void random_fill(StridedMemRefType_f32_1D *mr, unsigned seed) {
  srand(seed);
  long long N = mr->sizes[0];
  for (long long i = 0; i < N; i++) {
    mr->aligned[mr->offset + i * mr->strides[0]] =
        (float)((rand() % 2001) - 1000) / 100.0f;
  }
}
// Golden Model 
static void ref_vector_min(float *out, StridedMemRefType_f32_1D *A) {
    long long len = A->sizes[0];
    float min_val = A->aligned[A->offset];
    for (long long i = 1; i < len; i++) {
        float val = A->aligned[A->offset + i * A->strides[0]];
        if (val < min_val) {
            min_val = val;
        }
    }
    *out = min_val;
}

// Calcola l'errore tra due float
static double error_check_scalar(float x, float ref) {
    return fabs((double)x - (double)ref);
}

// Stampa il vettore (utile per debug)
static void print_vector(StridedMemRefType_f32_1D *vec, const char *name, int max_print) {
    printf("%s = [", name);
    long long N = vec->sizes[0];
    int print_count = (N < max_print) ? N : max_print;
    
    for (long long i = 0; i < print_count; i++) {
        printf("%s%.3f", i > 0 ? ", " : "", 
               vec->aligned[vec->offset + i * vec->strides[0]]);
    }
    if (N > max_print) {
        printf(", ...");
    }
    printf("]\n");
}

int main(int argc, char **argv) {
  long long N = 64; 
  if (argc == 2) { N = atoll(argv[1]); }

  srand(time(NULL));

  StridedMemRefType_f32_1D A = make_memref_1d(N);
  
  float ref_result = 0.0f;

  printf("=== INIZIO TEST VECTOR MIN (Dimensione: %lld) ===\n\n", N);

  for(int test = 1; test <= 6; test++) {
      
      random_fill(&A, 13 + test);
      
      printf("Test %d:\n", test);
      print_vector(&A, "  Input", 10);
      
      // Esecuzione MLIR: restituisce direttamente lo scalare minimo
      float mlir_result = _mlir_ciface_vector_min(&A);
      
      // Esecuzione Golden Standard in C
      ref_vector_min(&ref_result, &A);
      
      // Controllo dell'errore
      double err = error_check_scalar(mlir_result, ref_result);
      printf("  MLIR minimo: %.6f\n", mlir_result);
      printf("  Ref  minimo: %.6f\n", ref_result);
      printf("  Errore: %.6e\n", err);
      
      if (err < 1e-3) {
          printf("  Esito: SUPERATO, Errore = %e\n\n", err);
      } else {
          printf("  Esito: FALLITO, Errore = %e\n\n", err);
      }
  }

  free_memref(&A);
  
  return 0;
}