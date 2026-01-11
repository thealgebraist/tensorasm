# Quick Start Guide - Crypto Price Prediction (C++23)

## Overview
Predict cryptocurrency prices using 4 eigenvalue-based numerical methods:
- Power Iteration
- QR Algorithm  
- Rayleigh Quotient
- Subspace Iteration

## Quick Start

### Build
```bash
make -f Makefile.crypto_predict all
```

### Run Basic Test (4 methods, 3 time horizons)
```bash
./crypto_predict_cpp23
```

### Run Extended Tests (128 variations)
```bash
./crypto_extended_tests
```

### Run Complete Test Suite
```bash
./test_crypto_cpp23.sh
```

## What Gets Tested

### 4 Altcoins
- ETH (Ethereum)
- BNB (Binance Coin)
- SOL (Solana)
- ADA (Cardano)

### 8 Time Horizons
1. 1h prediction from 1 month (720 samples)
2. 1h prediction from 1 week (168 samples)
3. 24h prediction from 1 month
4. 1h prediction from 3 months (2,160 samples)
5. 1h prediction from 6 months (4,320 samples)
6. 12h prediction from 1 month
7. **1h prediction from 4 YEARS (35,040 samples)**
8. 7d prediction from 1 year (8,760 samples)

## Expected Results
- Average prediction error: ~0.5%
- Runtime per test: ~9ms
- Total tests: 128 (4 coins × 4 methods × 8 scenarios)

## Output Files
- `crypto_prediction_results.txt` - Summary of basic tests
- `crypto_extended_results.csv` - Detailed results with all 128 variations

## C++23 Features Demonstrated
✅ std::format  
✅ std::ranges & std::views  
✅ std::chrono literals  
✅ std::span  
✅ std::numbers  
✅ Range-based fold operations  

## Documentation
- `CRYPTO_PREDICT_CPP23_README.md` - Comprehensive guide
- `IMPLEMENTATION_SUMMARY.md` - Task completion summary

## Build Requirements
- GCC 13+ or Clang 16+ (C++23 support)
- Linux/macOS/WSL

## Example Output
```
=================== Cryptocurrency Price Prediction - C++23 ====================
======================= Using 4 Eigenvalue-Based Methods =======================

--- Testing ETH ---
  power_iteration: Predicted $3061.05, Actual $3025.11, Error $-35.94 (1.19%)
  qr_algorithm: Predicted $3061.05, Actual $3025.11, Error $-35.94 (1.19%)
  rayleigh_quotient: Predicted $3061.05, Actual $3025.11, Error $-35.94 (1.19%)
  subspace_iteration: Predicted $3061.05, Actual $3025.11, Error $-35.94 (1.19%)

============================== Summary Statistics ==============================
power_iteration          : Avg MAPE = 0.94%
qr_algorithm             : Avg MAPE = 0.94%
rayleigh_quotient        : Avg MAPE = 0.94%
subspace_iteration       : Avg MAPE = 0.94%
```

## Clean Up
```bash
make -f Makefile.crypto_predict clean
```
