func.func @mat_setall(%val: f32, %src: tensor<?x?xf32>) -> tensor<?x?xf32> 
attributes { llvm.emit_c_interface } {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  
  %dim_M = tensor.dim %src, %c0 : tensor<?x?xf32>
  %dim_N = tensor.dim %src, %c1 : tensor<?x?xf32>
  
  %init = tensor.empty(%dim_M, %dim_N) : tensor<?x?xf32>
  
  %res = linalg.fill ins(%val : f32) 
                     outs(%init : tensor<?x?xf32>) -> tensor<?x?xf32>
  
  return %res : tensor<?x?xf32>
}