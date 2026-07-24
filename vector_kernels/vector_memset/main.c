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


extern void _mlir_ciface_vector_memset(StridedMemRefType_f32_1D *A, 
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

// Riempie il vettore con valori casuali [-10, 10]
static void random_fill(StridedMemRefType_f32_1D *mr, unsigned seed) {
  srand(seed);
  long long N = mr->sizes[0];
  for (long long i = 0; i < N; i++) {
    mr->aligned[mr->offset + i * mr->strides[0]] =
        (float)((rand() % 2001) - 1000) / 100.0f;
  }
}

// Azzera il vettore (utile per la destinazione prima della copia)
static void zero_fill(StridedMemRefType_f32_1D *mr) {
  long long N = mr->sizes[0];
  for (long long i = 0; i < N; i++) {
    mr->aligned[mr->offset + i * mr->strides[0]] = 0.0f;
  }
}

// Golden Model 
static void ref_vector_memset(StridedMemRefType_f32_1D *out, StridedMemRefType_f32_1D *in) {
    long long len = in->sizes[0];
    for (long long idx = 0; idx < len; idx++) {
        out->aligned[out->offset + idx * out->strides[0]] = 
            in->aligned[in->offset + idx * in->strides[0]];
    }
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
    if (N > max_print) printf(", ...");
    printf("]\n");
}


// Calcola l'errore tra il risultato MLIR e il Golden Model
static double error_check(const StridedMemRefType_f32_1D *X,
                          const StridedMemRefType_f32_1D *Ref) {
  long long N = X->sizes[0];
  double s = 0.0;
  for (long long i = 0; i < N; i++) {
      double dx = (double)X->aligned[X->offset + i * X->strides[0]]
                - (double)Ref->aligned[Ref->offset + i * Ref->strides[0]];
      s += dx * dx;
  }
  return s;
}

int main(int argc, char **argv) {
  long long N = 64; 
  if (argc == 2) { N = atoll(argv[1]); }

  srand(time(NULL));

  StridedMemRefType_f32_1D B = make_memref_1d(N);       
  StridedMemRefType_f32_1D A_mlir = make_memref_1d(N);  
  StridedMemRefType_f32_1D A_ref = make_memref_1d(N);  

  printf("=== INIZIO TEST VECTOR COPY (Dimensione: %lld) ===\n\n", N);

  for(int test = 1; test <= 6; test++) {
      random_fill(&B, 13 + test); 
      zero_fill(&A_mlir);
      zero_fill(&A_ref);
      
      printf("Test %d:\n", test);
      print_vector(&B, "  Sorgente (B)    ", 8);
      
      // Esecuzione MLIR
      _mlir_ciface_vector_memset(&A_mlir, &B);
      
      // Esecuzione C standard
      ref_vector_memset(&A_ref, &B);
      
      // Controllo dell'errore sull'intero array
      print_vector(&A_mlir, "  Risultato MLIR  ", 8);
      print_vector(&A_ref, "  Risultato Golden ", 8);

      double max_err = error_check(&A_mlir, &A_ref);
      printf("  Errore massimo: %.6e\n", max_err);
      
      if (max_err < 1e-3) {
          printf("  Esito: SUPERATO, Errore = %e\n\n", max_err);
      } else {
          printf("  Esito: FALLITO, Errore = %e\n\n", max_err);
      }
  }

  free_memref(&B);
  free_memref(&A_mlir);
  free_memref(&A_ref);
  
  return 0;
}