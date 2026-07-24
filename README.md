# 🧬 Design and Optimization of Linear Algebra Kernels in MLIR

![MLIR](https://img.shields.io/badge/Framework-MLIR-blue.svg)
![LLVM](https://img.shields.io/badge/Backend-LLVM-success.svg)
![Arch](https://img.shields.io/badge/Architecture-x86%20%7C%20ARM%20%7C%20RISC--V-orange.svg)
![Status](https://img.shields.io/badge/Status-Completed-brightgreen.svg)


This repository contains the source code and experimental results of my thesis project: *"Design and optimization of linear algebra kernels for heterogeneous architectures based on the MLIR framework"*.

---

## 🎯 Project Objective

With the decline of Moore's Law and the worsening of the *memory wall*, the hardware landscape is shifting towards heterogeneous and specialized architectures (GPU, NPU, RISC-V). Traditional compilers struggle to scale with this complexity, leading to the "N to M" fragmentation problem.

This project explores a solution based on **MLIR (Multi-Level Intermediate Representation)**: expressing linear algebra algorithms through abstract and declarative constructs (`tensor` and `linalg` dialects), decoupling them from the underlying hardware architecture. This preserves the mathematical semantics of the operation, making the kernels modular, optimizable, and **portable without source code modifications across x86, ARM, and RISC-V**.

## 🧠 Key Concepts and MLIR Abstraction

- **`linalg` Dialect and Intrinsic Semantics:** Unlike traditional C, where matrix multiplication loses its mathematical meaning and becomes a simple triple `for` loop, `linalg.matmul` declares *what* to compute, not *how*. Iterations, indexing, and reductions are guaranteed by the operation itself.
- **Progressive Lowering:** The code passes through multiple levels of abstraction. Tiling, fusion, and parallelization are applied directly at a high level, before being lowered to the LLVM backend and specific hardware.
- **Integration with `scf`:** For algorithms with complex temporal dependencies (e.g., Cholesky Decomposition), where a pure *data-parallel* approach is not applicable, outer loops are handled using the imperative `scf` dialect, keeping the inner operations in `linalg`.

---

## 📂 Repository Structure

The library consists of about 20 kernels, categorized by complexity, all validated against *golden standards* implemented in C.

```text
.
├── vector_kernels/               # Basic vector operations
│   ├── vector_add, vector_dot, vector_scale, vector_min, etc.
├── matrix_kernels/               # Matrix operations
│   ├── mat_mul, mat_trans, mat_swaprow, etc.
├── advanced_algorithm_kernels/   # Complex linear algebra algorithms
│   ├── cholensky                 # Cholesky Decomposition
│   ├── lu_decomp & lu_solve      # LU decomposition and solver
│   ├── gemv & gemv_trans         # Generalized matrix-vector multiplication
│   └── svd & Jacobi_SVD          # Singular Value Decomposition
└── exec.sh                       # Automated execution script
---
Each kernel folder contains:

*.mlir: The abstract implementation of the algorithm in the MLIR framework.

main.c: The C reference implementation (golden standard) used for correctness testing.

test/: Logs and support files for validation.
# Run the bash script to validate the kernels
./exec.sh
(Ensure that the paths to mlir-opt, mlir-translate, and clang are correctly configured in your PATH or within the exec.sh script).
