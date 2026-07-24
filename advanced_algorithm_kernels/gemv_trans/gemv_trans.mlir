module {
  func.func @linalg_gemv_trans(
      %mat   : tensor<?x?xf32>,
      %vec_x : tensor<?xf32>,   
      %vec_y : tensor<?xf32>,   
      %alpha : f32,            
      %beta  : f32            
  ) -> tensor<?xf32> attributes { llvm.emit_c_interface } {

    %c1 = arith.constant 1 : index
    %zero = arith.constant 0.0 : f32

    %dim_N = tensor.dim %mat, %c1 : tensor<?x?xf32>

    %sum_empty = tensor.empty(%dim_N) : tensor<?xf32>
    %sum_init  = linalg.fill ins(%zero : f32) outs(%sum_empty : tensor<?xf32>) -> tensor<?xf32>

    %vecmat_res = linalg.vecmat 
        ins(%vec_x, %mat : tensor<?xf32>, tensor<?x?xf32>) 
        outs(%sum_init : tensor<?xf32>) -> tensor<?xf32>

    %dst_empty = tensor.empty(%dim_N) : tensor<?xf32>
    
    %dst = linalg.generic {
        indexing_maps = [
            affine_map<(d0) -> (d0)>,
            affine_map<(d0) -> (d0)>, 
            affine_map<(d0) -> (d0)>  
        ],
        iterator_types = ["parallel"]
    } ins(%vecmat_res, %vec_y : tensor<?xf32>, tensor<?xf32>)
      outs(%dst_empty : tensor<?xf32>) {
    ^bb0(%sum_val: f32, %y_val: f32, %out: f32):
        
        %alpha_sum = arith.mulf %alpha, %sum_val : f32
        %beta_y    = arith.mulf %beta, %y_val : f32
        %final_val = arith.addf %alpha_sum, %beta_y : f32
        
        linalg.yield %final_val : f32
    } -> tensor<?xf32>

    return %dst : tensor<?xf32>
  }
}