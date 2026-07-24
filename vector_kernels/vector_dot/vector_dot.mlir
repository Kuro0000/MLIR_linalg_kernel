module {
  func.func @vector_dot(%A: tensor<?xf32>,
                    %B: tensor<?xf32>) -> f32
      attributes { llvm.emit_c_interface } {

    %c0 = arith.constant 0 : index

    %m = tensor.dim %A, %c0 : tensor<?xf32>

    %n = tensor.dim %B, %c0 : tensor<?xf32>

    %zero_scalar = arith.constant 0.0 : f32
    %zero = tensor.from_elements %zero_scalar : tensor<f32>

    %C = linalg.dot ins(%A, %B : tensor<?xf32>, tensor<?xf32>) outs(%zero : tensor<f32>) -> tensor<f32>

    %result = tensor.extract %C[] : tensor<f32>

    return %result : f32
  }
}

