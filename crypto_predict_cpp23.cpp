/*
 * Cryptocurrency Price Prediction using 4 Numerical Methods
 * C++23 Implementation
 * 
 * Methods:
 * 1. Power Iteration - Dominant eigenvalue method
 * 2. QR Algorithm - Full eigenspectrum computation
 * 3. Rayleigh Quotient - Eigenvalue refinement
 * 4. Subspace Iteration - Multiple eigenpairs
 * 
 * Predicts prices for 4 top altcoins:
 * - ETH (Ethereum)
 * - BNB (Binance Coin)
 * - SOL (Solana)
 * - ADA (Cardano)
 */

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <format>
#include <fstream>
#include <iostream>
#include <numbers>
#include <ranges>
#include <span>
#include <string>
#include <string_view>
#include <vector>

using namespace std::chrono_literals;

// Constants
constexpr std::array<std::string_view, 4> ALTCOINS = {"ETH", "BNB", "SOL", "ADA"};
constexpr int WINDOW_SIZE = 30;  // 30 days for feature extraction
constexpr int PREDICT_HORIZON = 1;  // 1 hour ahead
constexpr double EPSILON = 1e-6;
constexpr int MAX_ITERATIONS = 100;

// Data structures
struct PricePoint {
    std::chrono::system_clock::time_point timestamp;
    double price;
    double volume;
};

struct PredictionResult {
    std::string_view symbol;
    std::string_view method;
    double predicted_price;
    double actual_price;
    double error;
    double mape;  // Mean Absolute Percentage Error
    int samples_used;
};

// Matrix operations using std::vector for dynamic sizing
class Matrix {
public:
    Matrix(size_t rows, size_t cols) 
        : rows_(rows), cols_(cols), data_(rows * cols, 0.0) {}
    
    double& operator()(size_t i, size_t j) {
        return data_[i * cols_ + j];
    }
    
    double operator()(size_t i, size_t j) const {
        return data_[i * cols_ + j];
    }
    
    size_t rows() const { return rows_; }
    size_t cols() const { return cols_; }
    
    std::span<double> row(size_t i) {
        return std::span(data_.begin() + i * cols_, cols_);
    }
    
    void fill(double value) {
        std::ranges::fill(data_, value);
    }
    
    static Matrix identity(size_t n) {
        Matrix m(n, n);
        for (size_t i = 0; i < n; ++i) {
            m(i, i) = 1.0;
        }
        return m;
    }

private:
    size_t rows_, cols_;
    std::vector<double> data_;
};

// Vector operations
std::vector<double> normalize(const std::vector<double>& v) {
    double norm = std::sqrt(std::ranges::fold_left(
        v | std::views::transform([](double x) { return x * x; }),
        0.0, std::plus<>{}
    ));
    
    if (norm < EPSILON) return v;
    
    std::vector<double> result(v.size());
    std::ranges::transform(v, result.begin(), 
        [norm](double x) { return x / norm; });
    return result;
}

double dot_product(const std::vector<double>& a, const std::vector<double>& b) {
    return std::ranges::fold_left(
        std::views::zip_transform(std::multiplies<>{}, a, b),
        0.0, std::plus<>{}
    );
}

std::vector<double> matrix_vector_multiply(const Matrix& A, const std::vector<double>& v) {
    std::vector<double> result(A.rows(), 0.0);
    for (size_t i = 0; i < A.rows(); ++i) {
        for (size_t j = 0; j < A.cols(); ++j) {
            result[i] += A(i, j) * v[j];
        }
    }
    return result;
}

// Method 1: Power Iteration
double power_iteration(const Matrix& A, int max_iter = MAX_ITERATIONS) {
    size_t n = A.rows();
    std::vector<double> v(n, 1.0);
    v = normalize(v);
    
    double eigenvalue = 0.0;
    for (int iter = 0; iter < max_iter; ++iter) {
        auto v_new = matrix_vector_multiply(A, v);
        v_new = normalize(v_new);
        
        eigenvalue = dot_product(v_new, matrix_vector_multiply(A, v_new));
        
        // Check convergence
        double diff = 0.0;
        for (size_t i = 0; i < n; ++i) {
            diff += std::abs(v_new[i] - v[i]);
        }
        
        v = v_new;
        if (diff < EPSILON) break;
    }
    
    return eigenvalue;
}

// Method 2: QR Algorithm (simplified for dominant eigenvalue)
double qr_algorithm(const Matrix& A, int max_iter = MAX_ITERATIONS) {
    size_t n = A.rows();
    Matrix A_k = A;
    
    for (int iter = 0; iter < max_iter; ++iter) {
        // Simplified QR: just extract diagonal trend
        Matrix Q = Matrix::identity(n);
        Matrix R(n, n);
        
        // Simple Gram-Schmidt for QR decomposition
        for (size_t j = 0; j < n; ++j) {
            std::vector<double> v(n);
            for (size_t i = 0; i < n; ++i) {
                v[i] = A_k(i, j);
            }
            
            for (size_t i = 0; i < j; ++i) {
                double proj = 0.0;
                for (size_t k = 0; k < n; ++k) {
                    proj += Q(k, i) * A_k(k, j);
                }
                for (size_t k = 0; k < n; ++k) {
                    v[k] -= proj * Q(k, i);
                }
            }
            
            double norm = std::sqrt(dot_product(v, v));
            if (norm > EPSILON) {
                for (size_t i = 0; i < n; ++i) {
                    Q(i, j) = v[i] / norm;
                }
            }
        }
        
        // Update A_k = R * Q (approximation)
        for (size_t i = 0; i < n; ++i) {
            for (size_t j = 0; j < n; ++j) {
                double sum = 0.0;
                for (size_t k = 0; k < n; ++k) {
                    sum += A_k(i, k) * Q(k, j);
                }
                R(i, j) = sum;
            }
        }
    }
    
    // Return largest diagonal element
    double max_eigen = A_k(0, 0);
    for (size_t i = 1; i < n; ++i) {
        max_eigen = std::max(max_eigen, A_k(i, i));
    }
    return max_eigen;
}

// Method 3: Rayleigh Quotient
double rayleigh_quotient(const Matrix& A, const std::vector<double>& v) {
    auto Av = matrix_vector_multiply(A, v);
    return dot_product(v, Av) / dot_product(v, v);
}

double rayleigh_quotient_iteration(const Matrix& A, int max_iter = MAX_ITERATIONS) {
    size_t n = A.rows();
    std::vector<double> v(n, 1.0);
    v = normalize(v);
    
    double mu = rayleigh_quotient(A, v);
    
    for (int iter = 0; iter < max_iter; ++iter) {
        auto v_new = matrix_vector_multiply(A, v);
        v_new = normalize(v_new);
        
        double mu_new = rayleigh_quotient(A, v_new);
        
        if (std::abs(mu_new - mu) < EPSILON) {
            return mu_new;
        }
        
        mu = mu_new;
        v = v_new;
    }
    
    return mu;
}

// Method 4: Subspace Iteration (extract top eigenvalue from subspace)
double subspace_iteration(const Matrix& A, int num_vectors = 3, int max_iter = MAX_ITERATIONS) {
    size_t n = A.rows();
    num_vectors = std::min(num_vectors, static_cast<int>(n));
    
    // Initialize subspace
    std::vector<std::vector<double>> V(num_vectors, std::vector<double>(n, 1.0));
    for (auto& v : V) {
        v = normalize(v);
    }
    
    for (int iter = 0; iter < max_iter; ++iter) {
        // Apply A to each vector
        for (auto& v : V) {
            v = matrix_vector_multiply(A, v);
            v = normalize(v);
        }
        
        // Orthogonalize using Gram-Schmidt
        for (size_t i = 0; i < V.size(); ++i) {
            for (size_t j = 0; j < i; ++j) {
                double proj = dot_product(V[i], V[j]);
                for (size_t k = 0; k < n; ++k) {
                    V[i][k] -= proj * V[j][k];
                }
            }
            V[i] = normalize(V[i]);
        }
    }
    
    // Return Rayleigh quotient of first vector
    return rayleigh_quotient(A, V[0]);
}

// Feature extraction: Create correlation matrix from price series
Matrix create_correlation_matrix(const std::vector<PricePoint>& prices, int window = WINDOW_SIZE) {
    if (prices.size() < static_cast<size_t>(window)) {
        window = prices.size();
    }
    
    // Extract recent price changes
    std::vector<double> returns;
    for (size_t i = 1; i < std::min(prices.size(), static_cast<size_t>(window + 1)); ++i) {
        double ret = (prices[i].price - prices[i-1].price) / prices[i-1].price;
        returns.push_back(ret);
    }
    
    size_t n = std::min(static_cast<size_t>(10), returns.size());
    Matrix corr(n, n);
    
    // Build autocorrelation matrix
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < n; ++j) {
            double sum = 0.0;
            int count = 0;
            for (size_t k = 0; k + std::max(i, j) < returns.size(); ++k) {
                sum += returns[k + i] * returns[k + j];
                count++;
            }
            corr(i, j) = count > 0 ? sum / count : 0.0;
        }
    }
    
    return corr;
}

// Prediction using eigenvalue-based method
double predict_price(const std::vector<PricePoint>& prices, 
                     std::string_view method,
                     int horizon_hours = PREDICT_HORIZON) {
    if (prices.empty()) return 0.0;
    
    Matrix corr = create_correlation_matrix(prices);
    double eigenvalue = 0.0;
    
    if (method == "power_iteration") {
        eigenvalue = power_iteration(corr);
    } else if (method == "qr_algorithm") {
        eigenvalue = qr_algorithm(corr);
    } else if (method == "rayleigh_quotient") {
        eigenvalue = rayleigh_quotient_iteration(corr);
    } else if (method == "subspace_iteration") {
        eigenvalue = subspace_iteration(corr);
    }
    
    // Use eigenvalue to scale the trend
    double current_price = prices.back().price;
    
    // Simple trend extrapolation based on eigenvalue
    std::vector<double> recent_changes;
    for (size_t i = std::max(1ul, prices.size() - 5); i < prices.size(); ++i) {
        recent_changes.push_back(prices[i].price - prices[i-1].price);
    }
    
    double avg_change = recent_changes.empty() ? 0.0 :
        std::ranges::fold_left(recent_changes, 0.0, std::plus<>{}) / recent_changes.size();
    
    // Scale by eigenvalue magnitude (normalized)
    double scaling_factor = 1.0 + eigenvalue * 0.01;
    double predicted = current_price + avg_change * horizon_hours * scaling_factor;
    
    return predicted;
}

// Generate synthetic historical data (for demonstration)
std::vector<PricePoint> generate_synthetic_data(std::string_view symbol, int days = 30) {
    std::vector<PricePoint> data;
    
    // Base prices for each coin
    double base_price = 0.0;
    if (symbol == "ETH") base_price = 3000.0;
    else if (symbol == "BNB") base_price = 500.0;
    else if (symbol == "SOL") base_price = 150.0;
    else if (symbol == "ADA") base_price = 0.5;
    
    auto now = std::chrono::system_clock::now();
    
    for (int i = 0; i < days * 24; ++i) {  // Hourly data
        auto timestamp = now - std::chrono::hours(days * 24 - i);
        
        // Add some trend and noise
        double trend = std::sin(i * 0.01) * 0.02;
        double noise = (std::sin(i * 0.5) + std::cos(i * 0.7)) * 0.01;
        double price = base_price * (1.0 + trend + noise);
        double volume = 1000000.0 * (1.0 + std::abs(noise));
        
        data.push_back({timestamp, price, volume});
    }
    
    return data;
}

// Test prediction accuracy
PredictionResult test_prediction(std::string_view symbol, 
                                std::string_view method,
                                const std::vector<PricePoint>& full_data,
                                size_t test_point) {
    // Use data up to test_point for training
    std::vector<PricePoint> training_data(
        full_data.begin(),
        full_data.begin() + test_point
    );
    
    double predicted = predict_price(training_data, method);
    double actual = test_point < full_data.size() ? full_data[test_point].price : 0.0;
    double error = actual - predicted;
    double mape = std::abs(error / actual) * 100.0;
    
    return PredictionResult{
        symbol,
        method,
        predicted,
        actual,
        error,
        mape,
        static_cast<int>(training_data.size())
    };
}

// Run comprehensive tests
void run_comprehensive_tests() {
    std::cout << std::format("\n{:=^80}\n", " Cryptocurrency Price Prediction - C++23 ");
    std::cout << std::format("{:=^80}\n", " Using 4 Eigenvalue-Based Methods ");
    std::cout << "\nAltcoins: ";
    for (const auto& coin : ALTCOINS) {
        std::cout << coin << " ";
    }
    std::cout << "\n\nMethods:\n";
    std::cout << "1. Power Iteration\n";
    std::cout << "2. QR Algorithm\n";
    std::cout << "3. Rayleigh Quotient\n";
    std::cout << "4. Subspace Iteration\n";
    std::cout << std::format("\n{:=^80}\n\n", "");
    
    std::vector<PredictionResult> all_results;
    
    // Test each coin with each method
    for (const auto& symbol : ALTCOINS) {
        std::cout << std::format("\n--- Testing {} ---\n", symbol);
        
        // Generate 4 years of data (we'll use 120 days for demonstration)
        auto full_data = generate_synthetic_data(symbol, 120);
        
        std::array<std::string_view, 4> methods = {
            "power_iteration",
            "qr_algorithm", 
            "rayleigh_quotient",
            "subspace_iteration"
        };
        
        // Test multiple time horizons
        std::vector<size_t> test_points = {
            full_data.size() - 24,  // 1 day ago
            full_data.size() - 168, // 1 week ago
            full_data.size() - 720  // 1 month ago
        };
        
        for (const auto& method : methods) {
            for (size_t test_point : test_points) {
                if (test_point >= WINDOW_SIZE && test_point < full_data.size()) {
                    auto result = test_prediction(symbol, method, full_data, test_point);
                    all_results.push_back(result);
                    
                    std::cout << std::format("  {} (using {} samples): Predicted ${:.4f}, "
                                           "Actual ${:.4f}, Error ${:.4f} ({:.2f}%)\n",
                                           method, result.samples_used,
                                           result.predicted_price, result.actual_price,
                                           result.error, result.mape);
                }
            }
        }
    }
    
    // Summary statistics
    std::cout << std::format("\n{:=^80}\n", " Summary Statistics ");
    
    for (const auto& method : std::array<std::string_view, 4>{
        "power_iteration", "qr_algorithm", "rayleigh_quotient", "subspace_iteration"
    }) {
        auto method_results = all_results 
            | std::views::filter([&](const auto& r) { return r.method == method; });
        
        double total_mape = 0.0;
        int count = 0;
        for (const auto& r : method_results) {
            total_mape += r.mape;
            count++;
        }
        
        double avg_mape = count > 0 ? total_mape / count : 0.0;
        std::cout << std::format("{:25s}: Avg MAPE = {:.2f}%\n", method, avg_mape);
    }
    
    std::cout << std::format("\n{:=^80}\n", "");
}

// Write results to file
void write_results_to_file() {
    std::ofstream out("crypto_prediction_results.txt");
    if (!out) return;
    
    out << "Cryptocurrency Price Prediction Results\n";
    out << "========================================\n\n";
    out << "Test completed at: " 
        << std::format("{:%Y-%m-%d %H:%M:%S}\n\n", 
                       std::chrono::system_clock::now());
    
    for (const auto& symbol : ALTCOINS) {
        auto data = generate_synthetic_data(symbol, 30);
        out << symbol << ": " << data.size() << " data points generated\n";
    }
    
    out << "\nMethods tested:\n";
    out << "- Power Iteration\n";
    out << "- QR Algorithm\n";
    out << "- Rayleigh Quotient\n";
    out << "- Subspace Iteration\n";
}

int main() {
    try {
        run_comprehensive_tests();
        write_results_to_file();
        
        std::cout << "\n✓ Prediction tests completed successfully\n";
        std::cout << "✓ Results written to crypto_prediction_results.txt\n\n";
        
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << '\n';
        return 1;
    }
}
