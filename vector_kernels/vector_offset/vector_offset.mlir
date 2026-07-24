module {
  func.func @vector_offset(%A: tensor<?xf32>,
                    %val: f32) -> tensor<?xf32>
      attributes { llvm.emit_c_interface } {

    %c0 = arith.constant 0 : index

    %m = tensor.dim %A, %c0 : tensor<?xf32>

    %empty   = tensor.empty(%m) : tensor<?xf32>
    %zero    = arith.constant 0.0 : f32
    %Czero   = linalg.fill ins(%zero : f32) outs(%empty : tensor<?xf32>) -> tensor<?xf32>

    %C = linalg.add ins(%A, %val : tensor<?xf32>, f32) outs(%Czero : tensor<?xf32>) -> tensor<?xf32>

    return %C : tensor<?xf32>
  }
}

