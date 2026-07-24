module {

  func.func @vector_memset(%B: tensor<?xf32>) -> tensor<?xf32> 
    attributes { llvm.emit_c_interface } {
    
    %c0 = arith.constant 0 : index
    %dim = tensor.dim %B, %c0 : tensor<?xf32>

    %init = tensor.empty(%dim) : tensor<?xf32>

    %result = linalg.copy 
      ins(%B : tensor<?xf32>) 
      outs(%init : tensor<?xf32>) -> tensor<?xf32>

    return %result : tensor<?xf32>
  }
}