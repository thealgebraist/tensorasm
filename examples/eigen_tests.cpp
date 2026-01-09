#include <iostream>
#include <Eigen/Dense>
#include <cmath>

using Mat = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
using Vec = Eigen::VectorXd;

static Mat make_symmetric(int n) {
    Mat B = Mat::Random(n, n);
    Mat A = B.transpose() * B; // SPD
    return A;
}

static double rayleigh(const Mat& A, const Vec& v) {
    return v.dot(A * v) / v.dot(v);
}

static double power_iteration(const Mat& A, Vec& v, int iters = 200) {
    v.normalize();
    for (int i = 0; i < iters; ++i) {
        v = A * v;
        v.normalize();
    }
    return rayleigh(A, v);
}

int main() {
    const int n = 128;
    Mat A = make_symmetric(n);

    // Ground truth via Eigen
    Eigen::SelfAdjointEigenSolver<Mat> es(A);
    Eigen::Index idx;
    auto evals = es.eigenvalues();
    double lambda_max_true_abs = evals.cwiseAbs().maxCoeff(&idx);
    double lambda_max_true = evals(idx);
    Vec v_true = es.eigenvectors().col(idx);

    // Power iteration approximation
    Vec v = Vec::Random(n);
    double lambda_pi = power_iteration(A, v, 2000);

    double rel_err = std::abs(std::abs(lambda_pi) - lambda_max_true_abs) / std::max(1.0, lambda_max_true_abs);
    std::cout << "lambda_true_abs=" << lambda_max_true_abs << ", lambda_pi_abs=" << std::abs(lambda_pi) << ", rel_err=" << rel_err << std::endl;
    if (rel_err > 1e-2) {
        std::cerr << "PowerIteration test FAILED" << std::endl;
        return 1;
    }

    // Rayleigh quotient test against direct computation
    Vec vr = Vec::Random(n);
    double rq = rayleigh(A, vr);
    double rq_direct = (vr.transpose() * A * vr)(0) / (vr.transpose() * vr)(0);
    double rq_err = std::abs(rq - rq_direct);
    std::cout << "rq=" << rq << ", rq_direct=" << rq_direct << ", err=" << rq_err << std::endl;
    if (rq_err > 1e-12) {
        std::cerr << "RayleighQuotient test FAILED" << std::endl;
        return 1;
    }

    // Compare eigenvector alignment (up to sign)
    double align = std::abs(v.dot(v_true));
    std::cout << "eigenvector alignment=" << align << std::endl;
    if (align < 0.9) {
        std::cerr << "Eigenvector alignment FAILED" << std::endl;
        return 1;
    }

    std::cout << "All eigen tests PASSED" << std::endl;
    return 0;
}
