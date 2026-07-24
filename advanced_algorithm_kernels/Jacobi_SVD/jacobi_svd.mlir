//
// Used dialects: tensor, linalg, arith, math, scf, func.
//

module {

  func.func @linalg_svd_jacobi(
      %mat_in : tensor<?x?xf32>
  ) -> (tensor<?x?xf32>, // diagonalised mat
        tensor<?x?xf32>, // right singular vectors
        tensor<?xf32>)   // singular values
  attributes { llvm.emit_c_interface } {

    %c0       = arith.constant 0       : index
    %c1       = arith.constant 1       : index
    %c0_i64   = arith.constant 0       : i64
    %c1_i64   = arith.constant 1       : i64
    %MAX_ITER = arith.constant 200     : i64
    %zero     = arith.constant 0.0     : f32
    %one      = arith.constant 1.0     : f32
    %two      = arith.constant 2.0     : f32
    %eps      = arith.constant 1.0e-12 : f32
    %inf      = arith.constant 1.0e38  : f32

    %M        = tensor.dim %mat_in, %c0 : tensor<?x?xf32>
    %M_minus1 = arith.subi %M, %c1 : index

    // ------------------------------------------------------------------
    // Init V with identity matrix
    // ------------------------------------------------------------------
    %V_empty = tensor.empty(%M, %M) : tensor<?x?xf32>
    %V_zero  = linalg.fill ins(%zero : f32)
                           outs(%V_empty : tensor<?x?xf32>) -> tensor<?x?xf32>
    %V_init  = linalg.generic {
        indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>],
        iterator_types = ["parallel", "parallel"]
    } outs(%V_zero : tensor<?x?xf32>) {
    ^bb0(%out : f32):
        %r       = linalg.index 0 : index
        %c_      = linalg.index 1 : index
        %is_diag = arith.cmpi eq, %r, %c_ : index
        %val     = arith.select %is_diag, %one, %zero : f32
        linalg.yield %val : f32
    } -> tensor<?x?xf32>

    // ------------------------------------------------------------------
    // Outer while loop
    // ------------------------------------------------------------------
    %_iter, %final_mat, %final_V, %_off =
      scf.while (%iter = %c0_i64, %mat = %mat_in, %V = %V_init, %max_off = %inf)
        : (i64, tensor<?x?xf32>, tensor<?x?xf32>, f32)-> (i64, tensor<?x?xf32>, tensor<?x?xf32>, f32) {

      %not_done       = arith.cmpi slt, %iter, %MAX_ITER : i64
      %not_converged  = arith.cmpf oge, %max_off, %eps : f32
      %keep_iterating = arith.andi %not_done, %not_converged : i1
      scf.condition(%keep_iterating) %iter, %mat, %V, %max_off
          : i64, tensor<?x?xf32>, tensor<?x?xf32>, f32

    } do {
    ^bb0(%iter : i64, %mat : tensor<?x?xf32>, %V : tensor<?x?xf32>, %max_off : f32):

      %i_mat, %i_V, %i_max =
        scf.for %i = %c0 to %M_minus1 step %c1
            iter_args(%imat = %mat, %iV = %V, %imax = %zero)
            -> (tensor<?x?xf32>, tensor<?x?xf32>, f32) {

          %i_plus1 = arith.addi %i, %c1 : index

          %j_mat, %j_V, %j_max =
            scf.for %j = %i_plus1 to %M step %c1
                iter_args(%jmat = %imat, %jV = %iV, %jmax = %imax)
                -> (tensor<?x?xf32>, tensor<?x?xf32>, f32) {

              // ----------------------------------------------------------
              // if (fabs(mat[i,j]) < eps) continue;
              // ----------------------------------------------------------
              %mij     = tensor.extract %jmat[%i, %j] : tensor<?x?xf32>
              %mij_abs = math.absf %mij : f32
              %do_rot  = arith.cmpf oge, %mij_abs, %eps : f32

              %new_mat, %new_V, %new_max =
                scf.if %do_rot -> (tensor<?x?xf32>, tensor<?x?xf32>, f32) {

                  // ------------------------------------------------------
                  // 1) Compute Givens rotation coefficients
                  // ------------------------------------------------------
                  %mii     = tensor.extract %jmat[%i, %i] : tensor<?x?xf32>
                  %mjj     = tensor.extract %jmat[%j, %j] : tensor<?x?xf32>

                  %diff    = arith.subf %mjj, %mii : f32
                  %two_mij = arith.mulf %two, %mij : f32
                  %tau     = arith.divf %diff, %two_mij : f32

                  %tau2    = arith.mulf %tau, %tau : f32
                  %one_t2  = arith.addf %one, %tau2 : f32
                  %sq_tau  = math.sqrt %one_t2 : f32

                  %tau_pos = arith.cmpf oge, %tau, %zero : f32
                  %dp      = arith.addf %tau, %sq_tau : f32
                  %dn      = arith.subf %tau, %sq_tau : f32
                  %t_denom = arith.select %tau_pos, %dp, %dn : f32
                  %t       = arith.divf %one, %t_denom : f32

                  %t2      = arith.mulf %t, %t : f32
                  %one_tt  = arith.addf %one, %t2 : f32
                  %sq_t    = math.sqrt %one_tt : f32
                  %cos_v   = arith.divf %one, %sq_t : f32
                  %sin_v   = arith.mulf %t, %cos_v : f32

                  // ------------------------------------------------------
                  // 2) Row update of mat
                  //   mat[i, m] = cos*mat[i,m] - sin*mat[j,m]
                  //   mat[j, m] = sin*mat[i,m] + cos*mat[j,m]
                  // outs(%jmat): every element overwritten, result aliases
                  // the iter arg so bufferization can prove equivalence.
                  // ------------------------------------------------------
                  %row_i = tensor.extract_slice %jmat[%i, 0][1, %M][1, 1]
                               : tensor<?x?xf32> to tensor<?xf32>
                  %row_j = tensor.extract_slice %jmat[%j, 0][1, %M][1, 1]
                               : tensor<?x?xf32> to tensor<?xf32>

                  %mat_rows = linalg.generic {
                      indexing_maps = [
                          affine_map<(d0, d1) -> (d0, d1)>,
                          affine_map<(d0, d1) -> (d1)>,
                          affine_map<(d0, d1) -> (d1)>,
                          affine_map<(d0, d1) -> (d0, d1)>
                      ],
                      iterator_types = ["parallel", "parallel"]
                  } ins(%jmat, %row_i, %row_j : tensor<?x?xf32>, tensor<?xf32>, tensor<?xf32>)
                    outs(%jmat : tensor<?x?xf32>) {
                  ^bb0(%orig : f32, %ri : f32, %rj : f32, %out : f32):
                      %r      = linalg.index 0 : index
                      %is_i   = arith.cmpi eq, %r, %i : index
                      %is_j   = arith.cmpi eq, %r, %j : index
                      %cos_ri = arith.mulf %cos_v, %ri : f32
                      %sin_rj = arith.mulf %sin_v, %rj : f32
                      %val_i  = arith.subf %cos_ri, %sin_rj : f32
                      %sin_ri = arith.mulf %sin_v, %ri : f32
                      %cos_rj = arith.mulf %cos_v, %rj : f32
                      %val_j  = arith.addf %sin_ri, %cos_rj : f32
                      %v      = arith.select %is_i, %val_i, %orig : f32
                      %v2     = arith.select %is_j, %val_j, %v   : f32
                      linalg.yield %v2 : f32
                  } -> tensor<?x?xf32>

                  // ------------------------------------------------------
                  // 3) Column update of mat
                  //   mat[n, i] = cos*mat[n,i] - sin*mat[n,j]
                  //   mat[n, j] = sin*mat[n,i] + cos*mat[n,j]
                  // ------------------------------------------------------
                  %col_i = tensor.extract_slice %mat_rows[0, %i][%M, 1][1, 1]
                               : tensor<?x?xf32> to tensor<?xf32>
                  %col_j = tensor.extract_slice %mat_rows[0, %j][%M, 1][1, 1]
                               : tensor<?x?xf32> to tensor<?xf32>

                  %mat_cols = linalg.generic {
                      indexing_maps = [
                          affine_map<(d0, d1) -> (d0, d1)>,
                          affine_map<(d0, d1) -> (d0)>,
                          affine_map<(d0, d1) -> (d0)>,
                          affine_map<(d0, d1) -> (d0, d1)>
                      ],
                      iterator_types = ["parallel", "parallel"]
                  } ins(%mat_rows, %col_i, %col_j : tensor<?x?xf32>, tensor<?xf32>, tensor<?xf32>)
                    outs(%mat_rows : tensor<?x?xf32>) {
                  ^bb0(%orig : f32, %ci : f32, %cj : f32, %out : f32):
                      %c_     = linalg.index 1 : index
                      %is_ci  = arith.cmpi eq, %c_, %i : index
                      %is_cj  = arith.cmpi eq, %c_, %j : index
                      %cos_ci = arith.mulf %cos_v, %ci : f32
                      %sin_cj = arith.mulf %sin_v, %cj : f32
                      %vci    = arith.subf %cos_ci, %sin_cj : f32
                      %sin_ci = arith.mulf %sin_v, %ci : f32
                      %cos_cj = arith.mulf %cos_v, %cj : f32
                      %vcj    = arith.addf %sin_ci, %cos_cj : f32
                      %v      = arith.select %is_ci, %vci, %orig : f32
                      %v2     = arith.select %is_cj, %vcj, %v   : f32
                      linalg.yield %v2 : f32
                  } -> tensor<?x?xf32>

                  // ------------------------------------------------------
                  // 4) Column update of V
                  //   V[n, i] = cos*V[n,i] - sin*V[n,j]
                  //   V[n, j] = sin*V[n,i] + cos*V[n,j]
                  // ------------------------------------------------------
                  %Vcol_i = tensor.extract_slice %jV[0, %i][%M, 1][1, 1]
                                : tensor<?x?xf32> to tensor<?xf32>
                  %Vcol_j = tensor.extract_slice %jV[0, %j][%M, 1][1, 1]
                                : tensor<?x?xf32> to tensor<?xf32>

                  %V_new = linalg.generic {
                      indexing_maps = [
                          affine_map<(d0, d1) -> (d0, d1)>,
                          affine_map<(d0, d1) -> (d0)>,
                          affine_map<(d0, d1) -> (d0)>,
                          affine_map<(d0, d1) -> (d0, d1)>
                      ],
                      iterator_types = ["parallel", "parallel"]
                  } ins(%jV, %Vcol_i, %Vcol_j : tensor<?x?xf32>, tensor<?xf32>, tensor<?xf32>)
                    outs(%jV : tensor<?x?xf32>) {
                  ^bb0(%orig : f32, %vi : f32, %vj : f32, %out : f32):
                      %c_     = linalg.index 1 : index
                      %is_ci  = arith.cmpi eq, %c_, %i : index
                      %is_cj  = arith.cmpi eq, %c_, %j : index
                      %cos_vi = arith.mulf %cos_v, %vi : f32
                      %sin_vj = arith.mulf %sin_v, %vj : f32
                      %vci    = arith.subf %cos_vi, %sin_vj : f32
                      %sin_vi = arith.mulf %sin_v, %vi : f32
                      %cos_vj = arith.mulf %cos_v, %vj : f32
                      %vcj    = arith.addf %sin_vi, %cos_vj : f32
                      %v      = arith.select %is_ci, %vci, %orig : f32
                      %v2     = arith.select %is_cj, %vcj, %v   : f32
                      linalg.yield %v2 : f32
                  } -> tensor<?x?xf32>

                  // ------------------------------------------------------
                  // 5) Accumulate max off-diagonal residual
                  // ------------------------------------------------------
                  %mij_post = tensor.extract %mat_cols[%i, %j] : tensor<?x?xf32>
                  %res_abs  = math.absf %mij_post : f32
                  %new_max  = arith.maximumf %jmax, %res_abs : f32

                  scf.yield %mat_cols, %V_new, %new_max : tensor<?x?xf32>, tensor<?x?xf32>, f32

                } else {
                  // fabs(mat[i,j]) < eps: skip, pass through unchanged
                  scf.yield %jmat, %jV, %jmax : tensor<?x?xf32>, tensor<?x?xf32>, f32
                }

              scf.yield %new_mat, %new_V, %new_max : tensor<?x?xf32>, tensor<?x?xf32>, f32
            } // end scf.for %j

          scf.yield %j_mat, %j_V, %j_max : tensor<?x?xf32>, tensor<?x?xf32>, f32
        } // end scf.for %i

      %iter_next = arith.addi %iter, %c1_i64 : i64
      scf.yield %iter_next, %i_mat, %i_V, %i_max : i64, tensor<?x?xf32>, tensor<?x?xf32>, f32
    } // end scf.while

    // ------------------------------------------------------------------
    // Extract singular values: vec_S[i] = sqrt(max(mat[i,i], 0))
    // ------------------------------------------------------------------
    %S_empty = tensor.empty(%M) : tensor<?xf32>
    %vec_S   = linalg.generic {
        indexing_maps = [
            affine_map<(d0) -> (d0, d0)>,
            affine_map<(d0) -> (d0)>
        ],
        iterator_types = ["parallel"]
    } ins(%final_mat : tensor<?x?xf32>)
      outs(%S_empty : tensor<?xf32>) {
    ^bb0(%diag : f32, %out : f32):
        %val = arith.maximumf %diag, %zero : f32
        %res = math.sqrt %val : f32
        linalg.yield %res : f32
    } -> tensor<?xf32>

    return %final_mat, %final_V, %vec_S : tensor<?x?xf32>, tensor<?x?xf32>, tensor<?xf32>
  }

}

