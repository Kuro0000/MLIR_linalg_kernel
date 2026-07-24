#!/bin/bash

ROOT_DIR=${1:-.}
> test.log
find "$ROOT_DIR" -type d | while read -r dir; do
    
    mlir_files=( "$dir"/*.mlir )
    if [ ! -e "${mlir_files[0]}" ] || [ ! -f "$dir/main.c" ]; then
        continue
    fi

    

    for mlir_path in "${mlir_files[@]}"; do
        filename=$(basename "$mlir_path" .mlir)
        ll_path="$dir/$filename.ll"
        output_exe="$dir/test"

        echo "Compilazione MLIR: $filename.mlir -> $filename.ll"
        mlir-opt "$mlir_path"   -canonicalize   -eliminate-empty-tensors   -linalg-generalize-named-ops   -one-shot-bufferize="bufferize-function-boundaries"   -buffer-deallocation-pipeline   -convert-bufferization-to-memref   -convert-linalg-to-loops   -convert-scf-to-cf   -expand-strided-metadata   -lower-affine   -arith-expand   -convert-cf-to-llvm   -convert-math-to-llvm   -convert-arith-to-llvm   -convert-index-to-llvm   -finalize-memref-to-llvm   -convert-func-to-llvm   -reconcile-unrealized-casts | mlir-translate --mlir-to-llvmir > "$ll_path"

        if [ $? -eq 0 ]; then
            echo "Linking con Clang: main.c + $filename.ll -> test"
            
            # 2. Compilazione con Clang
            clang "$dir/main.c" "$ll_path" \
                -L/usr/lib64 \
                -lmlir_runner_utils \
                -lmlir_c_runner_utils \
                -Wl,-rpath,/usr/lib64 \
                -lm \
                -o "$output_exe"

            if [ $? -eq 0 ]; then
                echo "SUCCESSO: Creato eseguibile '$output_exe'"
                echo "Esecuzione di '$output_exe'..." >> test.log
                $output_exe >> test.log 2>&1
                echo "\\" >> test.log
                echo "----------------------------------------" >> test.log
                echo "\\" >> test.log
                echo "Output di '$output_exe' aggiunto a test.log"
            else
                echo "ERRORE: Linking fallito per $filename"
            fi

            rm -f "$ll_path"
        else
            echo "ERRORE: Compilazione MLIR fallita per $filename"
        fi
    done
done
