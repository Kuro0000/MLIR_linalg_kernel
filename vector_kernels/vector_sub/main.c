#include <stdio.h>
#include <stdlib.h>

typedef struct {
  float *allocated;
  float *aligned;
  long long offset;
  long long sizes[1];     
  long long strides[1];  
} StridedMemRefType_f32_1D;

extern void _mlir_ciface_vector_sub(StridedMemRefType_f32_1D *out,
                                    StridedMemRefType_f32_1D *A,
                                    StridedMemRefType_f32_1D *B);

static StridedMemRefType_f32_1D make_memref_1d(long long N) {
  StridedMemRefType_f32_1D mr;
  size_t bytes = (size_t)N * sizeof(float);
  float *data = (float*)aligned_alloc(64, bytes);
  mr.allocated = data;
  mr.aligned   = data;
  mr.offset    = 0;
  mr.sizes[0]  = N;
  mr.strides[0]= 1;
  return mr;
}

static void free_memref(StridedMemRefType_f32_1D *mr) {
  free(mr->allocated);
  mr->allocated = mr->aligned = NULL;
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

// Riempie il vettore con zeri
static void zero_fill(StridedMemRefType_f32_1D *mr) {
  long long N = mr->sizes[0];
  for (long long i = 0; i < N; i++) {
    mr->aligned[mr->offset + i * mr->strides[0]] = 0.0f;
  }
}

//Golden Model 
static void ref_vector_sub(const StridedMemRefType_f32_1D *A,
                           const StridedMemRefType_f32_1D *B,
                           StridedMemRefType_f32_1D *C) {
    long long len = A->sizes[0];
    for (int i = 0; i < len; i++) {
        float a = A->aligned[A->offset + i * A->strides[0]];
        float b = B->aligned[B->offset + i * B->strides[0]];
        C->aligned[C->offset + i * C->strides[0]] = a - b;
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

  StridedMemRefType_f32_1D A = make_memref_1d(N);
  StridedMemRefType_f32_1D B = make_memref_1d(N);
  StridedMemRefType_f32_1D C = make_memref_1d(N);
  StridedMemRefType_f32_1D Cref = make_memref_1d(N);
    for(int i = 0; i < 4; i++) {
        printf("test %i\n", i+1);
        random_fill(&A, 13);
        random_fill(&B, 42);
        zero_fill(&C);
        zero_fill(&Cref);

        // Esecuzione MLIR
        _mlir_ciface_vector_sub(&C, &A, &B);
        
        // Esecuzione Golden Standard in C
        ref_vector_sub(&A, &B, &Cref);
        print_vector(&A, "Input A", 8);
        print_vector(&B, "Input B", 8);
        print_vector(&C, "Output MLIR", 8);
        print_vector(&Cref, "Output Golden", 8);
        // Controllo dell'errore
        double err = error_check(&C, &Cref);
        printf("Error: %.6e\n", err);
}
  free_memref(&A); free_memref(&B); free_memref(&C); free_memref(&Cref);
  
  return 0;
}