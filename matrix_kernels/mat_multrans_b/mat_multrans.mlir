module {
  func.func @matrix_mul_trans_b(
      %arg0: tensor<?x?xf32>, 
      %arg1: tensor<?x?xf32> 
  ) -> tensor<?x?xf32> attributes { llvm.emit_c_interface } {
    
    %c0 = arith.constant 0 : index
    

    %dim_M = tensor.dim %arg0, %c0 : tensor<?x?xf32>
    %dim_P = tensor.dim %arg1, %c0 : tensor<?x?xf32>

    %init = tensor.empty(%dim_M, %dim_P) : tensor<?x?xf32>

    %cst = arith.constant 0.0 : f32
    %zero_init = linalg.fill ins(%cst : f32) 
                             outs(%init : tensor<?x?xf32>) -> tensor<?x?xf32>

    %res = linalg.matmul
           indexing_maps = [
             affine_map<(m, n, k) -> (m, k)>,   // A normale
             affine_map<(m, n, k) -> (n, k)>,   // B trasposta: legge (n,k) invece di (k,n)
             affine_map<(m, n, k) -> (m, n)>    // C output
           ]
           ins(%arg0, %arg1 : tensor<?x?xf32>, tensor<?x?xf32>)
           outs(%zero_init : tensor<?x?xf32>) -> tensor<?x?xf32>

    return %res : tensor<?x?xf32>
  }
}