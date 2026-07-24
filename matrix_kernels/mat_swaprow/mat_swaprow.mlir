module {
  func.func @mat_swaprow(%A: tensor<?x?xf32>, %row_a: index, %row_b: index) -> tensor<?x?xf32> attributes { llvm.emit_c_interface } {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index

    %nRows = tensor.dim %A, %c0 : tensor<?x?xf32>
    %nCols = tensor.dim %A, %c1 : tensor<?x?xf32>

    %rowA = tensor.extract_slice %A[%row_a, %c0] [1, %nCols] [1, 1]
      : tensor<?x?xf32> to tensor<1x?xf32>
    %rowB = tensor.extract_slice %A[%row_b, %c0] [1, %nCols] [1, 1]
      : tensor<?x?xf32> to tensor<1x?xf32>

    %tmp = tensor.insert_slice %rowB into %A[%row_a, %c0] [1, %nCols] [1, 1]
      : tensor<1x?xf32> into tensor<?x?xf32>

    %res = tensor.insert_slice %rowA into %tmp[%row_b, %c0] [1, %nCols] [1, 1]
      : tensor<1x?xf32> into tensor<?x?xf32>

    return %res : tensor<?x?xf32>
  }
}