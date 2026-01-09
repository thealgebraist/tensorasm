#!/bin/bash
set -euo pipefail

EIGEN_INC=""
for dir in "/usr/local/include/eigen3" "/usr/include/eigen3" "/opt/homebrew/include/eigen3" "/usr/local/include" "/usr/include"; do
  if [ -d "$dir/Eigen" ]; then
    EIGEN_INC="$dir"
    break
  fi
done

if [ -z "$EIGEN_INC" ]; then
  echo "Eigen3 not found. Install Eigen3 or set EIGEN_INC."
  exit 1
fi

tmp=$(mktemp -d)
trap "rm -rf \"$tmp\"" EXIT

g++ -O3 -std=c++17 -I "$EIGEN_INC" tensorasm.cpp -o "$tmp/tensorasm"
g++ -O3 -std=c++17 -I "$EIGEN_INC" tensor_interpreter.cpp -o "$tmp/tensor_interpreter"

example="examples/hf_inference.ta"

"$tmp/tensor_interpreter" "$example" > "$tmp/interpreter.out"
"$tmp/tensorasm" "$example" > "$tmp/gen.cpp"
g++ -O3 -std=c++17 -I "$EIGEN_INC" "$tmp/gen.cpp" -o "$tmp/compiler.out.bin"
"$tmp/compiler.out.bin" > "$tmp/compiler.run.log"

grep -vE "Interpretation successful.|Kernel execution successful." "$tmp/interpreter.out" > "$tmp/interpreter.clean"
grep -vE "Interpretation successful.|Kernel execution successful." "$tmp/compiler.run.log" > "$tmp/compiler.clean"

echo "Interpreter output:"
cat "$tmp/interpreter.clean"
echo
echo "Compiler output:"
cat "$tmp/compiler.clean"
echo

if diff "$tmp/interpreter.clean" "$tmp/compiler.clean" > /dev/null; then
  echo "Outputs match."
else
  echo "Outputs diverge; see above for interpreter vs compiler." >&2
  exit 1
fi
