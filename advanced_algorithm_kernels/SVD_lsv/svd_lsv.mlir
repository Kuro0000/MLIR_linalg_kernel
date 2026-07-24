module {
  func.func @linalg_svd_lsv(
      %src   : tensor<?x?xf32>,
      %mat_V : tensor<?x?xf32>,
      %vec_S : tensor<?xf32>
  ) -> tensor<?x?xf32> attributes { llvm.emit_c_interface } {

    %c0       = arith.constant 0 : index
    %c1       = arith.constant 1 : index
    %zero     = arith.constant 0.0 : f32
    %epsilon  = arith.constant 1.0e-12 : f32 

    %dim_M = tensor.dim %src, %c0 : tensor<?x?xf32>
    %dim_N = tensor.dim %src, %c1 : tensor<?x?xf32>

    %T_init = tensor.empty(%dim_M, %dim_N) : tensor<?x?xf32>
    %T_zero = linalg.fill ins(%zero : f32) 
                          outs(%T_init : tensor<?x?xf32>) -> tensor<?x?xf32>

    %T_mat  = linalg.matmul 
              ins(%src, %mat_V : tensor<?x?xf32>, tensor<?x?xf32>)
              outs(%T_zero : tensor<?x?xf32>) -> tensor<?x?xf32>


    %dst_init = tensor.empty(%dim_M, %dim_N) : tensor<?x?xf32>
    
    %dst = linalg.generic {
        indexing_maps = [
            affine_map<(d_m, d_k) -> (d_m, d_k)>, // Lettura da %T_mat (2D)
            affine_map<(d_m, d_k) -> (d_k)>,      // Lettura da %vec_S (1D, proiettato sulle colonne)
            affine_map<(d_m, d_k) -> (d_m, d_k)>  // Scrittura in %dst (2D)
        ],
        iterator_types = ["parallel", "parallel"]
    } ins(%T_mat, %vec_S : tensor<?x?xf32>, tensor<?xf32>)
      outs(%dst_init : tensor<?x?xf32>) {
    ^bb0(%t_val : f32, %s_val : f32, %out : f32):
        
        %is_small = arith.cmpf olt, %s_val, %epsilon : f32
        
        %div_val = arith.divf %t_val, %s_val : f32
        
        %final_val = arith.select %is_small, %zero, %div_val : f32
        
        linalg.yield %final_val : f32
    } -> tensor<?x?xf32>

    return %dst : tensor<?x?xf32>
  }
}