; ModuleID = 'LLVMDialectModule'
source_filename = "LLVMDialectModule"

declare ptr @malloc(i64)

define { ptr, ptr, i64, [1 x i64], [1 x i64] } @vector_offset(ptr %0, ptr %1, i64 %2, i64 %3, i64 %4, float %5) {
  %7 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } poison, ptr %0, 0
  %8 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %7, ptr %1, 1
  %9 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %8, i64 %2, 2
  %10 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %9, i64 %3, 3, 0
  %11 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %10, i64 %4, 4, 0
  %12 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %11, 3
  %13 = alloca [1 x i64], i64 1, align 8
  store [1 x i64] %12, ptr %13, align 4
  %14 = getelementptr [1 x i64], ptr %13, i32 0, i64 0
  %15 = load i64, ptr %14, align 4
  %16 = getelementptr float, ptr null, i64 %15
  %17 = ptrtoint ptr %16 to i64
  %18 = add i64 %17, 64
  %19 = call ptr @malloc(i64 %18)
  %20 = ptrtoint ptr %19 to i64
  %21 = add i64 %20, 63
  %22 = urem i64 %21, 64
  %23 = sub i64 %21, %22
  %24 = inttoptr i64 %23 to ptr
  %25 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } poison, ptr %19, 0
  %26 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %25, ptr %24, 1
  %27 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %26, i64 0, 2
  %28 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %27, i64 %15, 3, 0
  %29 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %28, i64 1, 4, 0
  br label %30

30:                                               ; preds = %33, %6
  %31 = phi i64 [ %36, %33 ], [ 0, %6 ]
  %32 = icmp slt i64 %31, %15
  br i1 %32, label %33, label %37

33:                                               ; preds = %30
  %34 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %29, 1
  %35 = getelementptr inbounds nuw float, ptr %34, i64 %31
  store float 0.000000e+00, ptr %35, align 4
  %36 = add i64 %31, 1
  br label %30

37:                                               ; preds = %30
  %38 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %11, 3
  %39 = alloca [1 x i64], i64 1, align 8
  store [1 x i64] %38, ptr %39, align 4
  %40 = getelementptr [1 x i64], ptr %39, i32 0, i64 0
  %41 = load i64, ptr %40, align 4
  br label %42

42:                                               ; preds = %45, %37
  %43 = phi i64 [ %56, %45 ], [ 0, %37 ]
  %44 = icmp slt i64 %43, %41
  br i1 %44, label %45, label %57

45:                                               ; preds = %42
  %46 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %11, 1
  %47 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %11, 2
  %48 = getelementptr float, ptr %46, i64 %47
  %49 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %11, 4, 0
  %50 = mul nuw nsw i64 %43, %49
  %51 = getelementptr inbounds nuw float, ptr %48, i64 %50
  %52 = load float, ptr %51, align 4
  %53 = fadd float %52, %5
  %54 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %29, 1
  %55 = getelementptr inbounds nuw float, ptr %54, i64 %43
  store float %53, ptr %55, align 4
  %56 = add i64 %43, 1
  br label %42

57:                                               ; preds = %42
  ret { ptr, ptr, i64, [1 x i64], [1 x i64] } %29
}

define void @_mlir_ciface_vector_offset(ptr %0, ptr %1, float %2) {
  %4 = load { ptr, ptr, i64, [1 x i64], [1 x i64] }, ptr %1, align 8
  %5 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %4, 0
  %6 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %4, 1
  %7 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %4, 2
  %8 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %4, 3, 0
  %9 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %4, 4, 0
  %10 = call { ptr, ptr, i64, [1 x i64], [1 x i64] } @vector_offset(ptr %5, ptr %6, i64 %7, i64 %8, i64 %9, float %2)
  store { ptr, ptr, i64, [1 x i64], [1 x i64] } %10, ptr %0, align 8
  ret void
}

!llvm.module.flags = !{!0}

!0 = !{i32 2, !"Debug Info Version", i32 3}
