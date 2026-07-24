module {
  func.func @vector_min(%arg0: tensor<?xf32>) -> f32 attributes { llvm.emit_c_interface } {
    
    %cst_inf = arith.constant 0x7F800000 : f32 
    %dest_init = tensor.empty() : tensor<f32>

    %dest_filled = linalg.fill ins(%cst_inf : f32) outs(%dest_init : tensor<f32>) -> tensor<f32>

    %res = linalg.reduce 
      ins(%arg0 : tensor<?xf32>) 
      outs(%dest_filled : tensor<f32>) 
      dimensions = [0]
      (%in: f32, %acc: f32) {
        %min = arith.minnumf %in, %acc : f32
        linalg.yield %min : f32
      }

    %result = tensor.extract %res[] : tensor<f32>

    return %result : f32
  }
}