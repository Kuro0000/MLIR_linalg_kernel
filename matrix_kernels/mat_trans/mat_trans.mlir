module {

  func.func @mat_transpose(%src: tensor<?x?xf32>) -> tensor<?x?xf32> 
  attributes { llvm.emit_c_interface } {
    
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    
    %dim_M = tensor.dim %src, %c0 : tensor<?x?xf32>
    %dim_N = tensor.dim %src, %c1 : tensor<?x?xf32>
    
    %init = tensor.empty(%dim_N, %dim_M) : tensor<?x?xf32>
    
    %res = linalg.transpose ins(%src : tensor<?x?xf32>)
                            outs(%init : tensor<?x?xf32>) 
                            permutation = [1, 0]
                    
    return %res : tensor<?x?xf32>
  }
}