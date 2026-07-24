#include <stdio.h>
#include <stdlib.h>
#include <time.h> 

typedef struct {
  float *allocated;
  float *aligned;
  long long offset;
  long long sizes[1];
  long long strides[1];
} StridedMemRefType_f32_1D;

extern void _mlir_ciface_vector_set(StridedMemRefType_f32_1D *out,
                                    StridedMemRefType_f32_1D *A,
                                    float val);

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
  free(mr->allocated);
  mr->allocated = mr->aligned = NULL;
}

// Riempie il vettore con zeri (utile per pulire l'output prima del test)
static void zero_fill(StridedMemRefType_f32_1D *mr) {
  long long N = mr->sizes[0];
  for (long long i = 0; i < N; i++) {
    mr->aligned[mr->offset + i * mr->strides[0]] = 0.0f;
  }
}

// Golden Model 
static void ref_vector_set(StridedMemRefType_f32_1D *C, float val) {
    long long len = C->sizes[0];
    for (int i = 0; i < len; i++) {
        C->aligned[C->offset + i * C->strides[0]] = val;
    }
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
  StridedMemRefType_f32_1D C = make_memref_1d(N);
  StridedMemRefType_f32_1D Cref = make_memref_1d(N);

  printf("=== INIZIO TEST VECTOR SET (Dimensione: %lld) ===\n\n", N);

  for(int test = 0; test <= 3; test++) {
      // Puliamo gli array di output prima di ogni test
      zero_fill(&C);
      zero_fill(&Cref);

      // Generiamo un numero float randomico tra -10.0 e 10.0
      float random_val = (float)((rand() % 2001) - 1000) / 100.0f;
      
      printf("Test %d: Imposto tutto il vettore a [%.3f]\n", test, random_val);

      // Esecuzione MLIR passando il valore scalare
      _mlir_ciface_vector_set(&C, &A, random_val);
      
      // Esecuzione Golden Standard in C passando lo stesso valore scalare
      ref_vector_set(&Cref, random_val);
      
      print_vector(&C, "C (MLIR)", 8);
      print_vector(&Cref, "Cref (Golden)", 8);

      // Controllo dell'errore tra MLIR (C) e Golden Model (Cref)
      double err = error_check(&C, &Cref);
      printf("Errore: %.6e\n", err);
      
      if (err < 1e-3) {
          printf("Esito: SUPERATO\n\n");
      } else {
          printf("Esito: FALLITO\n\n");
      }
  }

  free_memref(&A); free_memref(&C); free_memref(&Cref);
  
  return 0;
}