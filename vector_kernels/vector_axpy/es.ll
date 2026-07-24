; ModuleID = 'LLVMDialectModule'
source_filename = "LLVMDialectModule"

declare ptr @malloc(i64)

define { ptr, ptr, i64, [1 x i64], [1 x i64] } @vector_axpy(ptr %0, ptr %1, i64 %2, i64 %3, i64 %4, ptr %5, ptr %6, i64 %7, i64 %8, i64 %9, float %10) {
  %12 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } poison, ptr %5, 0
  %13 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %12, ptr %6, 1
  %14 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %13, i64 %7, 2
  %15 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %14, i64 %8, 3, 0
  %16 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %15, i64 %9, 4, 0
  %17 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } poison, ptr %0, 0
  %18 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %17, ptr %1, 1
  %19 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %18, i64 %2, 2
  %20 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %19, i64 %3, 3, 0
  %21 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %20, i64 %4, 4, 0
  %22 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %21, 3
  %23 = alloca [1 x i64], i64 1, align 8
  store [1 x i64] %22, ptr %23, align 4
  %24 = getelementptr [1 x i64], ptr %23, i32 0, i64 0
  %25 = load i64, ptr %24, align 4
  %26 = getelementptr float, ptr null, i64 %25
  %27 = ptrtoint ptr %26 to i64
  %28 = add i64 %27, 64
  %29 = call ptr @malloc(i64 %28)
  %30 = ptrtoint ptr %29 to i64
  %31 = add i64 %30, 63
  %32 = urem i64 %31, 64
  %33 = sub i64 %31, %32
  %34 = inttoptr i64 %33 to ptr
  %35 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } poison, ptr %29, 0
  %36 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %35, ptr %34, 1
  %37 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %36, i64 0, 2
  %38 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %37, i64 %25, 3, 0
  %39 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %38, i64 1, 4, 0
  %40 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %21, 3
  %41 = alloca [1 x i64], i64 1, align 8
  store [1 x i64] %40, ptr %41, align 4
  %42 = getelementptr [1 x i64], ptr %41, i32 0, i64 0
  %43 = load i64, ptr %42, align 4
  br label %44

44:                                               ; preds = %47, %11
  %45 = phi i64 [ %66, %47 ], [ 0, %11 ]
  %46 = icmp slt i64 %45, %43
  br i1 %46, label %47, label %67

47:                                               ; preds = %44
  %48 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %21, 1
  %49 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %21, 2
  %50 = getelementptr float, ptr %48, i64 %49
  %51 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %21, 4, 0
  %52 = mul nuw nsw i64 %45, %51
  %53 = getelementptr inbounds nuw float, ptr %50, i64 %52
  %54 = load float, ptr %53, align 4
  %55 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %16, 1
  %56 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %16, 2
  %57 = getelementptr float, ptr %55, i64 %56
  %58 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %16, 4, 0
  %59 = mul nuw nsw i64 %45, %58
  %60 = getelementptr inbounds nuw float, ptr %57, i64 %59
  %61 = load float, ptr %60, align 4
  %62 = fmul float %54, %10
  %63 = fadd float %62, %61
  %64 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %39, 1
  %65 = getelementptr inbounds nuw float, ptr %64, i64 %45
  store float %63, ptr %65, align 4
  %66 = add i64 %45, 1
  br label %44

67:                                               ; preds = %44
  ret { ptr, ptr, i64, [1 x i64], [1 x i64] } %39
}

define void @_mlir_ciface_vector_axpy(ptr %0, ptr %1, ptr %2, float %3) {
  %5 = load { ptr, ptr, i64, [1 x i64], [1 x i64] }, ptr %1, align 8
  %6 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %5, 0
  %7 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %5, 1
  %8 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %5, 2
  %9 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %5, 3, 0
  %10 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %5, 4, 0
  %11 = load { ptr, ptr, i64, [1 x i64], [1 x i64] }, ptr %2, align 8
  %12 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %11, 0
  %13 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %11, 1
  %14 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %11, 2
  %15 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %11, 3, 0
  %16 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %11, 4, 0
  %17 = call { ptr, ptr, i64, [1 x i64], [1 x i64] } @vector_axpy(ptr %6, ptr %7, i64 %8, i64 %9, i64 %10, ptr %12, ptr %13, i64 %14, i64 %15, i64 %16, float %3)
  store { ptr, ptr, i64, [1 x i64], [1 x i64] } %17, ptr %0, align 8
  ret void
}

!llvm.module.flags = !{!0}

!0 = !{i32 2, !"Debug Info Version", i32 3}
