module {
  func.func @linalg_lu_solve(
      %mat  : tensor<?x?xf32>,
      %vec  : tensor<?xf32>, 
      %perm : tensor<?xi32>    
  ) -> tensor<?xf32> attributes { llvm.emit_c_interface } {

    %c0   = arith.constant 0 : index
    %c1   = arith.constant 1 : index
    %zero = arith.constant 0.0 : f32

    %dim = tensor.dim %mat, %c0 : tensor<?x?xf32>
    
    %y_init = tensor.empty(%dim) : tensor<?xf32>
    
    %res_empty = tensor.empty(%dim) : tensor<?xf32>
    %res_init  = linalg.fill ins(%zero : f32) outs(%res_empty : tensor<?xf32>) -> tensor<?xf32>
    %acc_t = tensor.empty() : tensor<f32>

    %y_final = scf.for %m = %c0 to %dim step %c1 iter_args(%curr_y = %y_init) -> tensor<?xf32> {
        %idx_i32 = tensor.extract %perm[%m] : tensor<?xi32>
        %idx = arith.index_cast %idx_i32 : i32 to index
        %vec_val = tensor.extract %vec[%idx] : tensor<?xf32>

        %has_prev = arith.cmpi sgt, %m, %c0 : index
        %sum = scf.if %has_prev -> f32 {
            
            %row_m = tensor.extract_slice %mat[%m, 0] [1, %m] [1, 1] : tensor<?x?xf32> to tensor<?xf32>
            %prev= tensor.extract_slice %curr_y[0] [%m] [1] : tensor<?xf32> to tensor<?xf32>          
            %acc_zero = linalg.fill ins(%zero : f32) outs(%acc_t : tensor<f32>) -> tensor<f32>
            %dot_res = linalg.dot ins(%row_m, %prev : tensor<?xf32>, tensor<?xf32>) outs(%acc_zero : tensor<f32>) -> tensor<f32>            
            %s = tensor.extract %dot_res[] : tensor<f32>
            scf.yield %s : f32
        } else {
            scf.yield %zero : f32
        }

        %y_m = arith.subf %vec_val, %sum : f32
        %y_next= tensor.insert %y_m into %curr_y[%m] : tensor<?xf32>

        scf.yield %y_next : tensor<?xf32>
    }


    %x_final = scf.for %i = %c0 to %dim step %c1 iter_args(%curr_res = %res_init) -> tensor<?xf32> {
        %dim_minus = arith.subi %dim, %c1 : index
        %m = arith.subi %dim_minus, %i : index        
        %m_plus = arith.addi %m, %c1 : index
        %col = arith.subi %dim, %m_plus : index

        %has_next = arith.cmpi sgt, %col, %c0 : index
        %sum = scf.if %has_next -> f32 {
            
            %row_m= tensor.extract_slice %mat[%m, %m_plus] [1, %col] [1, 1] : tensor<?x?xf32> to tensor<?xf32>
            %res_next= tensor.extract_slice %curr_res[%m_plus] [%col] [1] : tensor<?xf32> to tensor<?xf32>
            %acc_zero = linalg.fill ins(%zero : f32) outs(%acc_t : tensor<f32>) -> tensor<f32>
            %dot_res= linalg.dot ins(%row_m, %res_next : tensor<?xf32>, tensor<?xf32>)   outs(%acc_zero : tensor<f32>) -> tensor<f32>           
            %s = tensor.extract %dot_res[] : tensor<f32>
            scf.yield %s : f32

        } else {
            scf.yield %zero : f32
        }
        %val = tensor.extract %y_final[%m] : tensor<?xf32>
        %mat_m = tensor.extract %mat[%m, %m] : tensor<?x?xf32>      
        %diff= arith.subf %val, %sum : f32
        %res_m = arith.divf %diff, %mat_m : f32      
        %res_next_iter= tensor.insert %res_m into %curr_res[%m] : tensor<?xf32>

        scf.yield %res_next_iter : tensor<?xf32>
    }

    return %x_final : tensor<?xf32>
  }


}
