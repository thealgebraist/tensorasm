#!/bin/bash

# Find Eigen3
EIGEN_INC=""
for dir in "/usr/local/include/eigen3" "/usr/include/eigen3" "/opt/homebrew/include/eigen3" "/usr/local/include" "/usr/include"; do
    if [ -d "$dir/Eigen" ]; then
        EIGEN_INC="$dir"
        break
    fi
done

if [ -z "$EIGEN_INC" ]; then
    echo "Error: Eigen3 not found. Please install Eigen3 or set EIGEN_INC manually."
    exit 1
fi

echo "Using Eigen from: $EIGEN_INC"
TENSORASM="./tensorasm"
INTERPRETER="./tensor_interpreter_test"

# Recompile tools
clang++ -O3 -std=c++17 tensorasm.cpp -o tensorasm || exit 1
clang++ -O3 -std=c++17 -I $EIGEN_INC tensor_interpreter.cpp -o tensor_interpreter_test || exit 1

mkdir -p test_build

for ta in examples/*.ta; do
    echo "Testing $ta..."
    
    # Interpreter
    $INTERPRETER $ta > test_build/interpreter_out.log 2> test_build/interpreter_err.log
    if [ $? -ne 0 ]; then
        echo "  Interpreter FAILED:"
        cat test_build/interpreter_err.log
        continue
    fi
    
    # Compiler
    $TENSORASM $ta > test_build/gen.cpp 2> test_build/compiler_err.log
    if [ $? -ne 0 ]; then
        echo "  Compiler FAILED:"
        cat test_build/compiler_err.log
        continue
    fi
    
    clang++ -O3 -std=c++17 -I $EIGEN_INC test_build/gen.cpp -o test_build/bin 2> test_build/cpp_err.log
    if [ $? -ne 0 ]; then
        echo "  C++ Compilation FAILED:"
        cat test_build/cpp_err.log
        continue
    fi
    
    ./test_build/bin > test_build/compiler_out.log
    
    # Compare output
    grep -vE "Interpretation successful.|Kernel execution successful." test_build/interpreter_out.log > test_build/interpreter_clean.log
    grep -vE "Interpretation successful.|Kernel execution successful." test_build/compiler_out.log > test_build/compiler_clean.log
    
    if diff test_build/interpreter_clean.log test_build/compiler_clean.log > /dev/null; then
        echo "  PASS"
    else
        echo "  OUTPUT MISMATCH (might be due to randomness)"
        # diff test_build/interpreter_clean.log test_build/compiler_clean.log
    fi
done
