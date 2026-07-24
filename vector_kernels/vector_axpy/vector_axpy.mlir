module {
  func.func @vector_axpy(
    %src_a: tensor<?xf32>, 
    %src_b: tensor<?xf32>, 
    %alpha: f32
  ) -> tensor<?xf32> attributes { llvm.emit_c_interface } {
    
    %c0 = arith.constant 0 : index
    %dim = tensor.dim %src_a, %c0 : tensor<?xf32>

    %init = tensor.empty(%dim) : tensor<?xf32>

    %res = linalg.map 
      ins(%src_a, %src_b : tensor<?xf32>, tensor<?xf32>) 
      outs(%init : tensor<?xf32>)
      (%val_a: f32, %val_b: f32, %out: f32) {
        %prod = arith.mulf %val_a, %alpha : f32
        %sum  = arith.addf %prod, %val_b : f32
        linalg.yield %sum : f32
      }

    return %res : tensor<?xf32>
  }
}