func.func @linalg_lu_decomp(%mat_in: tensor<?x?xf32>) -> (tensor<?x?xf32>, tensor<?xi32>) attributes { llvm.emit_c_interface } {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index

  %dim_m = tensor.dim %mat_in, %c0 : tensor<?x?xf32>
  %dim_n = tensor.dim %mat_in, %c1 : tensor<?x?xf32>

  %perm_empty = tensor.empty(%dim_m) : tensor<?xi32>
  %perm_init = linalg.generic {
    indexing_maps = [affine_map<(d0) -> (d0)>],
    iterator_types = ["parallel"]
  } outs(%perm_empty : tensor<?xi32>) {
  ^bb0(%out: i32):
    %idx = linalg.index 0 : index
    %idx_i32 = arith.index_cast %idx : index to i32
    linalg.yield %idx_i32 : i32
  } -> tensor<?xi32>

  %is_m_less = arith.cmpi slt, %dim_m, %dim_n : index
  %min_dim = arith.select %is_m_less, %dim_m, %dim_n : index

  %res_mat, %res_perm = scf.for %k = %c0 to %min_dim step %c1
      iter_args(%mat = %mat_in, %perm = %perm_init) -> (tensor<?x?xf32>, tensor<?xi32>) {

    %k_plus_1 = arith.addi %k, %c1 : index

    // 1. Pivot search
    %piv_init_val = tensor.extract %mat[%k, %k] : tensor<?x?xf32>
    %piv_init_abs = math.absf %piv_init_val : f32

    %max_row, %max_abs = scf.for %i = %k_plus_1 to %dim_m step %c1
        iter_args(%best_row = %k, %best_abs = %piv_init_abs) -> (index, f32) {
      %curr_val = tensor.extract %mat[%i, %k] : tensor<?x?xf32>
      %curr_abs = math.absf %curr_val : f32
      %is_greater = arith.cmpf ogt, %curr_abs, %best_abs : f32

      %next_row = arith.select %is_greater, %i, %best_row : index
      %next_abs = arith.select %is_greater, %curr_abs, %best_abs : f32
      scf.yield %next_row, %next_abs : index, f32
    }

    %needs_swap = arith.cmpi ne, %max_row, %k : index
    %mat_swapped, %perm_swapped = scf.if %needs_swap -> (tensor<?x?xf32>, tensor<?xi32>) {
      // Slice extraction
      %row_k = tensor.extract_slice %mat[%k, 0] [1, %dim_n] [1, 1] : tensor<?x?xf32> to tensor<?xf32>
      %row_max = tensor.extract_slice %mat[%max_row, 0] [1, %dim_n] [1, 1] : tensor<?x?xf32> to tensor<?xf32>

      // Matrix swap
      %mat_upd1 = tensor.insert_slice %row_max into %mat[%k, 0] [1, %dim_n] [1, 1] : tensor<?xf32> into tensor<?x?xf32>
      %mat_upd2 = tensor.insert_slice %row_k into %mat_upd1[%max_row, 0] [1, %dim_n] [1, 1] : tensor<?xf32> into tensor<?x?xf32>

      // Permutation vector swap
      %p_k = tensor.extract %perm[%k] : tensor<?xi32>
      %p_max = tensor.extract %perm[%max_row] : tensor<?xi32>
      %perm_upd1 = tensor.insert %p_max into %perm[%k] : tensor<?xi32>
      %perm_upd2 = tensor.insert %p_k into %perm_upd1[%max_row] : tensor<?xi32>

      scf.yield %mat_upd2, %perm_upd2 : tensor<?x?xf32>, tensor<?xi32>
    } else {
      scf.yield %mat, %perm : tensor<?x?xf32>, tensor<?xi32>
    }

    %pivot_val = tensor.extract %mat_swapped[%k, %k] : tensor<?x?xf32>
    %rows_left = arith.subi %dim_m, %k_plus_1 : index
    %cols_left = arith.subi %dim_n, %k_plus_1 : index

    %has_rows = arith.cmpi sgt, %rows_left, %c0 : index
    %mat_next = scf.if %has_rows -> tensor<?x?xf32> {

      %l_col_slice = tensor.extract_slice %mat_swapped[%k_plus_1, %k] [%rows_left, 1] [1, 1] : tensor<?x?xf32> to tensor<?xf32>
      %l_col_empty = tensor.empty(%rows_left) : tensor<?xf32>

      %l_col = linalg.generic {
        indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>],
        iterator_types = ["parallel"]
      } ins(%l_col_slice : tensor<?xf32>) outs(%l_col_empty : tensor<?xf32>) {
      ^bb0(%val: f32, %out: f32):
        %scaled = arith.divf %val, %pivot_val : f32
        linalg.yield %scaled : f32
      } -> tensor<?xf32>

      %mat_l = tensor.insert_slice %l_col into %mat_swapped[%k_plus_1, %k] [%rows_left, 1] [1, 1] : tensor<?xf32> into tensor<?x?xf32>

      %has_cols = arith.cmpi sgt, %cols_left, %c0 : index
      %mat_final = scf.if %has_cols -> tensor<?x?xf32> {
        %u_row = tensor.extract_slice %mat_l[%k, %k_plus_1] [1, %cols_left] [1, 1] : tensor<?x?xf32> to tensor<?xf32>
        %submat = tensor.extract_slice %mat_l[%k_plus_1, %k_plus_1] [%rows_left, %cols_left] [1, 1] : tensor<?x?xf32> to tensor<?x?xf32>
        %submat_empty = tensor.empty(%rows_left, %cols_left) : tensor<?x?xf32>

        %submat_upd = linalg.generic {
          indexing_maps = [
            affine_map<(d0, d1) -> (d0, d1)>,
            affine_map<(d0, d1) -> (d0)>,
            affine_map<(d0, d1) -> (d1)>,
            affine_map<(d0, d1) -> (d0, d1)>
          ],
          iterator_types = ["parallel", "parallel"]
        } ins(%submat, %l_col, %u_row : tensor<?x?xf32>, tensor<?xf32>, tensor<?xf32>)
          outs(%submat_empty : tensor<?x?xf32>) {
        ^bb0(%s: f32, %l: f32, %u: f32, %out: f32):
          %prod = arith.mulf %l, %u : f32
          %diff = arith.subf %s, %prod : f32
          linalg.yield %diff : f32
        } -> tensor<?x?xf32>

        %mat_u = tensor.insert_slice %submat_upd into %mat_l[%k_plus_1, %k_plus_1] [%rows_left, %cols_left] [1, 1] : tensor<?x?xf32> into tensor<?x?xf32>
        scf.yield %mat_u : tensor<?x?xf32>
      } else {
        scf.yield %mat_l : tensor<?x?xf32>
      }
      scf.yield %mat_final : tensor<?x?xf32>
    } else {
      scf.yield %mat_swapped : tensor<?x?xf32>
    }

    scf.yield %mat_next, %perm_swapped : tensor<?x?xf32>, tensor<?xi32>
  }

  return %res_mat, %res_perm : tensor<?x?xf32>, tensor<?xi32>
}
