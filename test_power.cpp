#include <iostream>
#include <Eigen/Dense>
#include "power_iteration_eigen.cpp"

int main() {
    std::cout << "Starting Power Iteration test with Eigen..." << std::endl;

    Matrix A; A.setRandom();
    Vector v; v.setRandom();
    Vector out; out.setZero();

    std::cout << "Input vector norm: " << v.norm() << std::endl;

    PowerIteration(A, v, out);

    std::cout << "Output vector norm: " << out.norm() << std::endl;
    std::cout << "Kernel execution finished successfully." << std::endl;

    return 0;
}
