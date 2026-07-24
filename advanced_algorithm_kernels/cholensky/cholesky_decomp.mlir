module {
  func.func @linalg_cholesky_decomp(
      %src: tensor<?x?xf32>
  ) -> tensor<?x?xf32> attributes { llvm.emit_c_interface } {

    %c0   = arith.constant 0 : index
    %c1   = arith.constant 1 : index
    %zero = arith.constant 0.0 : f32
    %one  = arith.constant 1.0 : f32

    %dim = tensor.dim %src, %c0 : tensor<?x?xf32>

    %dst_init = tensor.empty(%dim, %dim) : tensor<?x?xf32>
    %dst_zero = linalg.fill ins(%zero : f32) outs(%dst_init : tensor<?x?xf32>) -> tensor<?x?xf32>
    %sum_0 = tensor.empty() : tensor<f32>
    %vec_empty = tensor.empty(%dim) : tensor<?xf32> 
    %final_dst = scf.for %m = %c0 to %dim step %c1 
        iter_args(%curr_dst = %dst_zero) -> tensor<?x?xf32> {
        
        %src_mm = tensor.extract %src[%m, %m] : tensor<?x?xf32>
        
        %row_m = tensor.extract_slice %curr_dst[%m, 0] [1, %m] [1, 1] 
                    : tensor<?x?xf32> to tensor<?xf32>

        %sum_init = linalg.fill ins(%zero : f32) outs(%sum_0 : tensor<f32>) -> tensor<f32>
        %sum_dot = linalg.dot ins(%row_m, %row_m : tensor<?xf32>, tensor<?xf32>) 
                              outs(%sum_init : tensor<f32>) -> tensor<f32>
        %sum_scalar = tensor.extract %sum_dot[] : tensor<f32>

        %diff_diag = arith.subf %src_mm, %sum_scalar : f32
        %dst_mm    = math.sqrt %diff_diag : f32
        %dst_diag  = tensor.insert %dst_mm into %curr_dst[%m, %m] : tensor<?x?xf32>

        %m_plus_1 = arith.addi %m, %c1 : index
        %col_size = arith.subi %dim, %m_plus_1 : index

        %has_col_elements = arith.cmpi sgt, %col_size, %c0 : index
        %dst_next = scf.if %has_col_elements -> tensor<?x?xf32> {
            
            %src_col = tensor.extract_slice %src[%m_plus_1, %m] [%col_size, 1] [1, 1] 
                         : tensor<?x?xf32> to tensor<?xf32>
            
            %mat_prev = tensor.extract_slice %dst_diag[%m_plus_1, 0] [%col_size, %m] [1, 1] 
                         : tensor<?x?xf32> to tensor<?x?xf32>
        %vec_slice = tensor.extract_slice %vec_empty[0] [%col_size] [1] : tensor<?xf32> to tensor<?xf32>
            %vec_zero  = linalg.fill ins(%zero : f32) outs(%vec_slice : tensor<?xf32>) -> tensor<?xf32>
            %sum_vec   = linalg.matvec ins(%mat_prev, %row_m : tensor<?x?xf32>, tensor<?xf32>) 
                                       outs(%vec_zero : tensor<?xf32>) -> tensor<?xf32>

            %col_res = linalg.generic {
                indexing_maps = [
                    affine_map<(d0) -> (d0)>, 
                    affine_map<(d0) -> (d0)>, 
                    affine_map<(d0) -> (d0)> 
                ],
                iterator_types = ["parallel"]
            } ins(%src_col, %sum_vec : tensor<?xf32>, tensor<?xf32>) 
              outs(%vec_slice : tensor<?xf32>) {
            ^bb0(%s_val: f32, %sum_val: f32, %out: f32):
                %diff = arith.subf %s_val, %sum_val : f32
                %res  = arith.divf %diff, %dst_mm : f32
                linalg.yield %res : f32
            } -> tensor<?xf32>

            %dst_updated = tensor.insert_slice %col_res 
                into %dst_diag[%m_plus_1, %m] [%col_size, 1] [1, 1] 
                : tensor<?xf32> into tensor<?x?xf32>
            
            scf.yield %dst_updated : tensor<?x?xf32>

        } else {
            scf.yield %dst_diag : tensor<?x?xf32>
        }

        scf.yield %dst_next : tensor<?x?xf32>
    }

    return %final_dst : tensor<?x?xf32>
  }
}
