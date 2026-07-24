; ModuleID = 'LLVMDialectModule'
source_filename = "LLVMDialectModule"

declare ptr @malloc(i64)

define { ptr, ptr, i64, [2 x i64], [2 x i64] } @matrix_mul_trans_a(ptr %0, ptr %1, i64 %2, i64 %3, i64 %4, i64 %5, i64 %6, ptr %7, ptr %8, i64 %9, i64 %10, i64 %11, i64 %12, i64 %13) {
  %15 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } poison, ptr %7, 0
  %16 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %15, ptr %8, 1
  %17 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %16, i64 %9, 2
  %18 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %17, i64 %10, 3, 0
  %19 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %18, i64 %12, 4, 0
  %20 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %19, i64 %11, 3, 1
  %21 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %20, i64 %13, 4, 1
  %22 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } poison, ptr %0, 0
  %23 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %22, ptr %1, 1
  %24 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %23, i64 %2, 2
  %25 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %24, i64 %3, 3, 0
  %26 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %25, i64 %5, 4, 0
  %27 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %26, i64 %4, 3, 1
  %28 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %27, i64 %6, 4, 1
  %29 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %28, 3
  %30 = alloca [2 x i64], i64 1, align 8
  store [2 x i64] %29, ptr %30, align 4
  %31 = getelementptr [2 x i64], ptr %30, i32 0, i64 0
  %32 = load i64, ptr %31, align 4
  %33 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %21, 3
  %34 = alloca [2 x i64], i64 1, align 8
  store [2 x i64] %33, ptr %34, align 4
  %35 = getelementptr [2 x i64], ptr %34, i32 0, i64 0
  %36 = load i64, ptr %35, align 4
  %37 = mul i64 %36, %32
  %38 = getelementptr float, ptr null, i64 %37
  %39 = ptrtoint ptr %38 to i64
  %40 = add i64 %39, 64
  %41 = call ptr @malloc(i64 %40)
  %42 = ptrtoint ptr %41 to i64
  %43 = add i64 %42, 63
  %44 = urem i64 %43, 64
  %45 = sub i64 %43, %44
  %46 = inttoptr i64 %45 to ptr
  %47 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } poison, ptr %41, 0
  %48 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %47, ptr %46, 1
  %49 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %48, i64 0, 2
  %50 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %49, i64 %32, 3, 0
  %51 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %50, i64 %36, 3, 1
  %52 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %51, i64 %36, 4, 0
  %53 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %52, i64 1, 4, 1
  br label %54

54:                                               ; preds = %68, %14
  %55 = phi i64 [ %69, %68 ], [ 0, %14 ]
  %56 = icmp slt i64 %55, %32
  br i1 %56, label %57, label %70

57:                                               ; preds = %54
  br label %58

58:                                               ; preds = %61, %57
  %59 = phi i64 [ %67, %61 ], [ 0, %57 ]
  %60 = icmp slt i64 %59, %36
  br i1 %60, label %61, label %68

61:                                               ; preds = %58
  %62 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %53, 1
  %63 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %53, 4, 0
  %64 = mul nuw nsw i64 %55, %63
  %65 = add nuw nsw i64 %64, %59
  %66 = getelementptr inbounds nuw float, ptr %62, i64 %65
  store float 0.000000e+00, ptr %66, align 4
  %67 = add i64 %59, 1
  br label %58

68:                                               ; preds = %58
  %69 = add i64 %55, 1
  br label %54

70:                                               ; preds = %54
  %71 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %28, 3
  %72 = alloca [2 x i64], i64 1, align 8
  store [2 x i64] %71, ptr %72, align 4
  %73 = getelementptr [2 x i64], ptr %72, i32 0, i64 0
  %74 = load i64, ptr %73, align 4
  %75 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %28, 3
  %76 = alloca [2 x i64], i64 1, align 8
  store [2 x i64] %75, ptr %76, align 4
  %77 = getelementptr [2 x i64], ptr %76, i32 0, i64 1
  %78 = load i64, ptr %77, align 4
  %79 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %21, 3
  %80 = alloca [2 x i64], i64 1, align 8
  store [2 x i64] %79, ptr %80, align 4
  %81 = getelementptr [2 x i64], ptr %80, i32 0, i64 1
  %82 = load i64, ptr %81, align 4
  br label %83

83:                                               ; preds = %131, %70
  %84 = phi i64 [ %132, %131 ], [ 0, %70 ]
  %85 = icmp slt i64 %84, %78
  br i1 %85, label %86, label %133

86:                                               ; preds = %83
  br label %87

87:                                               ; preds = %129, %86
  %88 = phi i64 [ %130, %129 ], [ 0, %86 ]
  %89 = icmp slt i64 %88, %82
  br i1 %89, label %90, label %131

90:                                               ; preds = %87
  br label %91

91:                                               ; preds = %94, %90
  %92 = phi i64 [ %128, %94 ], [ 0, %90 ]
  %93 = icmp slt i64 %92, %74
  br i1 %93, label %94, label %129

94:                                               ; preds = %91
  %95 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %28, 1
  %96 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %28, 2
  %97 = getelementptr float, ptr %95, i64 %96
  %98 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %28, 4, 0
  %99 = mul nuw nsw i64 %92, %98
  %100 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %28, 4, 1
  %101 = mul nuw nsw i64 %84, %100
  %102 = add nuw nsw i64 %99, %101
  %103 = getelementptr inbounds nuw float, ptr %97, i64 %102
  %104 = load float, ptr %103, align 4
  %105 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %21, 1
  %106 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %21, 2
  %107 = getelementptr float, ptr %105, i64 %106
  %108 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %21, 4, 0
  %109 = mul nuw nsw i64 %92, %108
  %110 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %21, 4, 1
  %111 = mul nuw nsw i64 %88, %110
  %112 = add nuw nsw i64 %109, %111
  %113 = getelementptr inbounds nuw float, ptr %107, i64 %112
  %114 = load float, ptr %113, align 4
  %115 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %53, 1
  %116 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %53, 4, 0
  %117 = mul nuw nsw i64 %84, %116
  %118 = add nuw nsw i64 %117, %88
  %119 = getelementptr inbounds nuw float, ptr %115, i64 %118
  %120 = load float, ptr %119, align 4
  %121 = fmul float %104, %114
  %122 = fadd float %120, %121
  %123 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %53, 1
  %124 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %53, 4, 0
  %125 = mul nuw nsw i64 %84, %124
  %126 = add nuw nsw i64 %125, %88
  %127 = getelementptr inbounds nuw float, ptr %123, i64 %126
  store float %122, ptr %127, align 4
  %128 = add i64 %92, 1
  br label %91

129:                                              ; preds = %91
  %130 = add i64 %88, 1
  br label %87

131:                                              ; preds = %87
  %132 = add i64 %84, 1
  br label %83

133:                                              ; preds = %83
  ret { ptr, ptr, i64, [2 x i64], [2 x i64] } %53
}

define void @_mlir_ciface_matrix_mul_trans_a(ptr %0, ptr %1, ptr %2) {
  %4 = load { ptr, ptr, i64, [2 x i64], [2 x i64] }, ptr %1, align 8
  %5 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %4, 0
  %6 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %4, 1
  %7 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %4, 2
  %8 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %4, 3, 0
  %9 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %4, 3, 1
  %10 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %4, 4, 0
  %11 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %4, 4, 1
  %12 = load { ptr, ptr, i64, [2 x i64], [2 x i64] }, ptr %2, align 8
  %13 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %12, 0
  %14 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %12, 1
  %15 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %12, 2
  %16 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %12, 3, 0
  %17 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %12, 3, 1
  %18 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %12, 4, 0
  %19 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %12, 4, 1
  %20 = call { ptr, ptr, i64, [2 x i64], [2 x i64] } @matrix_mul_trans_a(ptr %5, ptr %6, i64 %7, i64 %8, i64 %9, i64 %10, i64 %11, ptr %13, ptr %14, i64 %15, i64 %16, i64 %17, i64 %18, i64 %19)
  store { ptr, ptr, i64, [2 x i64], [2 x i64] } %20, ptr %0, align 8
  ret void
}

!llvm.module.flags = !{!0}

!0 = !{i32 2, !"Debug Info Version", i32 3}
