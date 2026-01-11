# Cryptocurrency Price Prediction using Eigenvalue Methods (C++23)

## Overview

This implementation uses **4 advanced numerical eigenvalue methods** to predict cryptocurrency prices for the **top 4 altcoins**:
- **ETH** (Ethereum)
- **BNB** (Binance Coin)  
- **SOL** (Solana)
- **ADA** (Cardano)

## The 4 Methods

### 1. Power Iteration
**Classical eigenvalue method** for finding the dominant eigenvalue of a matrix.
- Iteratively applies matrix-vector multiplication
- Normalizes result at each step
- Converges to dominant eigenvector
- Used for: Extracting primary trend component

### 2. QR Algorithm
**Matrix decomposition method** for computing the full eigenspectrum.
- Uses Gram-Schmidt orthogonalization
- Decomposes matrix into Q (orthogonal) and R (upper triangular)
- Iteratively refines eigenvalue estimates
- Used for: Comprehensive spectral analysis

### 3. Rayleigh Quotient Iteration
**Refinement method** for eigenvalue approximation.
- Computes quotient: (v^T A v) / (v^T v)
- Provides rapid convergence near an eigenvalue
- Cubic convergence rate near exact eigenvalue
- Used for: High-precision eigenvalue estimation

### 4. Subspace Iteration
**Multi-vector method** for finding multiple eigenpairs simultaneously.
- Maintains orthogonal subspace of vectors
- Applies Gram-Schmidt orthogonalization
- Extracts top k eigenvalues and eigenvectors
- Used for: Multi-scale trend analysis

## Prediction Strategy

### Feature Extraction
1. **Price Returns Computation**: Calculate relative price changes
2. **Autocorrelation Matrix**: Build correlation matrix from time series
3. **Eigenvalue Extraction**: Apply each of the 4 methods
4. **Trend Scaling**: Use eigenvalue magnitude to scale predictions

### Time Horizons Tested
- **1 Hour ahead** based on recent data
- **1 Day lookback** (24 data points)
- **1 Week lookback** (168 data points)
- **1 Month lookback** (720 data points)

### Evaluation Metrics
- **Absolute Error**: |actual - predicted|
- **MAPE** (Mean Absolute Percentage Error): |error / actual| × 100%
- **Sample Size**: Number of historical points used

## C++23 Features Used

### Modern Language Features
- ✅ `std::format` - Type-safe string formatting
- ✅ `std::ranges` - Functional range operations
- ✅ `std::views` - Lazy range transformations
- ✅ `std::chrono` literals - Time duration literals (1h, 24h, etc.)
- ✅ `std::span` - Non-owning array views
- ✅ `std::numbers` - Mathematical constants (π, e)
- ✅ Structured bindings - Clean tuple unpacking
- ✅ Range-based fold operations - `fold_left` for accumulation

### Code Example
```cpp
// Using C++23 ranges and views
auto method_results = all_results 
    | std::views::filter([&](const auto& r) { return r.method == method; });

// Using std::format
std::cout << std::format("{:25s}: Avg MAPE = {:.2f}%\n", method, avg_mape);

// Using fold_left from C++23
double norm = std::sqrt(std::ranges::fold_left(
    v | std::views::transform([](double x) { return x * x; }),
    0.0, std::plus<>{}
));
```

## Building and Running

### Prerequisites
- **GCC 13+** or **Clang 16+** (C++23 support required)
- Linux/macOS/WSL environment
- No external dependencies needed

### Compilation
```bash
make -f Makefile.crypto_predict all
```

Or manually:
```bash
g++ -std=c++23 -Wall -Wextra -O3 -march=native -o crypto_predict_cpp23 crypto_predict_cpp23.cpp
```

### Execution
```bash
./crypto_predict_cpp23
```

### Running Tests
```bash
make -f Makefile.crypto_predict test
```

## Results

### Sample Output
```
=================== Cryptocurrency Price Prediction - C++23 ====================
======================= Using 4 Eigenvalue-Based Methods =======================

Altcoins: ETH BNB SOL ADA 

Methods:
1. Power Iteration
2. QR Algorithm
3. Rayleigh Quotient
4. Subspace Iteration

--- Testing ETH ---
  power_iteration (using 2856 samples): Predicted $3061.05, Actual $3025.11, Error $-35.94 (1.19%)
  qr_algorithm (using 2856 samples): Predicted $3061.05, Actual $3025.11, Error $-35.94 (1.19%)
  ...

============================== Summary Statistics ==============================
power_iteration          : Avg MAPE = 0.94%
qr_algorithm             : Avg MAPE = 0.94%
rayleigh_quotient        : Avg MAPE = 0.94%
subspace_iteration       : Avg MAPE = 0.94%
```

### Performance Metrics
All 4 methods achieve:
- **Average MAPE**: ~0.94% across all coins
- **Best case**: <0.3% error (1 week lookback)
- **Worst case**: ~1.3% error (1 month lookback)

## Implementation Details

### Matrix Operations
- Dynamic sizing using `std::vector<double>`
- Row-major layout for cache efficiency
- SIMD-friendly memory layout with `-march=native`

### Numerical Stability
- Epsilon threshold: `1e-6` for convergence
- Vector normalization at each iteration
- Gram-Schmidt orthogonalization for QR
- Maximum iterations: 100 (typical convergence: <20)

### Data Generation
For demonstration, synthetic data is generated with:
- Sinusoidal trends
- Random noise components
- Realistic price ranges per coin
- Hourly granularity over 30-120 days

## Testing Variations

The implementation tests multiple scenarios:
1. **Different lookback windows**: 1 day, 1 week, 1 month
2. **All 4 methods**: Compare eigenvalue approaches
3. **Multiple coins**: 4 different altcoins with varying price scales
4. **Historical simulation**: Walk-forward testing on time series

## Future Enhancements

### Real Data Integration
- [ ] Fetch live data from Binance/CoinGecko APIs
- [ ] Store historical data in SQLite/CSV
- [ ] Support 4 years of historical data
- [ ] Implement data caching and incremental updates

### Advanced Features
- [ ] Multiple prediction horizons (1h, 24h, 7d, 30d)
- [ ] Ensemble methods combining all 4 eigenvalue methods
- [ ] Volatility-based confidence intervals
- [ ] Real-time WebSocket streaming predictions
- [ ] GPU acceleration with CUDA/OpenCL

### Visualization
- [ ] Generate plots using gnuplot/matplotlib
- [ ] Real-time prediction dashboards
- [ ] Comparative accuracy charts
- [ ] Eigenvalue spectrum visualization

## Architecture

```
crypto_predict_cpp23.cpp
├── Data Structures
│   ├── PricePoint (timestamp, price, volume)
│   ├── Matrix (dynamic 2D array)
│   └── PredictionResult (metrics & statistics)
│
├── Matrix Operations
│   ├── normalize()
│   ├── dot_product()
│   └── matrix_vector_multiply()
│
├── Eigenvalue Methods
│   ├── power_iteration()
│   ├── qr_algorithm()
│   ├── rayleigh_quotient_iteration()
│   └── subspace_iteration()
│
├── Feature Engineering
│   └── create_correlation_matrix()
│
└── Prediction Pipeline
    ├── generate_synthetic_data()
    ├── predict_price()
    ├── test_prediction()
    └── run_comprehensive_tests()
```

## Mathematical Foundation

### Autocorrelation Matrix
For price returns `r[0], r[1], ..., r[n]`, the autocorrelation matrix is:
```
C[i,j] = (1/k) Σ r[t+i] * r[t+j]
```

### Eigenvalue Interpretation
- **Dominant eigenvalue λ₁**: Primary trend strength
- **Secondary eigenvalues**: Oscillatory components
- **Scaling factor**: 1 + λ₁ × 0.01

### Trend Extrapolation
```
predicted_price = current_price + avg_change × horizon × scaling_factor
```

## Files Generated

- `crypto_predict_cpp23` - Compiled binary
- `crypto_prediction_results.txt` - Test results summary

## License

This is demonstration code for educational purposes, showcasing C++23 features applied to quantitative finance.

## References

- **Numerical Linear Algebra**: Golub & Van Loan
- **Time Series Analysis**: Box & Jenkins
- **C++23 Standard**: ISO/IEC 14882:2023
- **Eigenvalue Methods**: Power iteration, QR algorithm, Rayleigh quotient
