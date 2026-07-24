module {
  func.func @vector_set(%vec: tensor<?xf32>, %val: f32) -> tensor<?xf32>
      attributes { llvm.emit_c_interface } {
    
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %len = tensor.dim %vec, %c0 : tensor<?xf32>
    %empty = tensor.empty(%len) : tensor<?xf32>
    
    %result = linalg.fill ins(%val : f32) outs(%empty : tensor<?xf32>) -> tensor<?xf32>
    
    return %result : tensor<?xf32>
  }
}