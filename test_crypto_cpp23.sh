#!/bin/bash
# Comprehensive test script for C++23 cryptocurrency price prediction
# Tests all 4 eigenvalue methods with various time horizons

set -e

echo "=========================================="
echo "Crypto Prediction C++23 Test Suite"
echo "Testing 4 Eigenvalue Methods"
echo "=========================================="
echo ""

# Check for C++23 compiler
if ! command -v g++ &> /dev/null; then
    echo "Error: g++ compiler not found"
    exit 1
fi

GCC_VERSION=$(g++ --version | head -1)
echo "Compiler: $GCC_VERSION"
echo ""

# Build all targets
echo "Building programs..."
make -f Makefile.crypto_predict all

echo ""
echo "=========================================="
echo "Test 1: Basic Prediction (4 methods)"
echo "=========================================="
./crypto_predict_cpp23

echo ""
echo "=========================================="
echo "Test 2: Extended Tests (128 variations)"
echo "=========================================="
./crypto_extended_tests

echo ""
echo "=========================================="
echo "Generated Files:"
echo "=========================================="
if [ -f crypto_prediction_results.txt ]; then
    echo "✓ crypto_prediction_results.txt ($(wc -l < crypto_prediction_results.txt) lines)"
fi

if [ -f crypto_extended_results.csv ]; then
    echo "✓ crypto_extended_results.csv ($(wc -l < crypto_extended_results.csv) lines)"
fi

echo ""
echo "=========================================="
echo "Summary of Extended Results:"
echo "=========================================="
if [ -f crypto_extended_results.csv ]; then
    echo ""
    echo "Method comparison (average error %):"
    for method in "power_iteration" "qr_algorithm" "rayleigh_quotient" "subspace_iteration"; do
        avg=$(awk -F',' -v m="$method" '$2 == m {sum+=$7; count++} END {if(count>0) print sum/count; else print "N/A"}' crypto_extended_results.csv)
        printf "  %-25s: %.2f%%\n" "$method" "$avg"
    done
    
    echo ""
    echo "Top 3 best predictions (lowest error):"
    tail -n +2 crypto_extended_results.csv | sort -t',' -k7 -n | head -3 | \
        awk -F',' '{printf "  %s: %s on %s (%.2f%% error)\n", $1, $2, $3, $7}'
    
    echo ""
    echo "Time horizon comparison:"
    echo "  1h predictions:     $(grep -c '1h prediction' crypto_extended_results.csv) tests"
    echo "  12h predictions:    $(grep -c '12h prediction' crypto_extended_results.csv) tests"
    echo "  24h predictions:    $(grep -c '24h prediction' crypto_extended_results.csv) tests"
    echo "  7d predictions:     $(grep -c '7d prediction' crypto_extended_results.csv) tests"
fi

echo ""
echo "=========================================="
echo "All Tests Completed Successfully! ✓"
echo "=========================================="
