module {
  func.func @vector_mul(%A: tensor<?xf32>,
                    %B: tensor<?xf32>) -> tensor<?xf32>
      attributes { llvm.emit_c_interface } {

    %c0 = arith.constant 0 : index

    %m = tensor.dim %A, %c0 : tensor<?xf32>

    %empty   = tensor.empty(%m) : tensor<?xf32>

    %C = linalg.mul ins(%A, %B : tensor<?xf32>, tensor<?xf32>) outs(%empty : tensor<?xf32>) -> tensor<?xf32>

    return %C : tensor<?xf32>
  }
}

