#include <Eigen/Eigen>
#include <iostream>

#include "Eigen/src/Core/Matrix.h"
#include "sophus/se3.hpp"

Eigen::Vector3d cost_func(Eigen::Vector3d zi, Eigen::Vector3d pi, Eigen::Matrix3d R) { return zi - R * pi; }

Eigen::Matrix3d J(Eigen::Vector3d pi, Eigen::Matrix3d R) { return -Sophus::SO3d::hat(R * pi); }

int main() {
  // Prepare data.
  std::vector<Eigen::Vector3d> zi = {{1, 2, 3}, {6, 3, 5}};
  std::vector<Eigen::Vector3d> pi = {{1, 2, 3}, {6, 3, 5}};

  // Initialize data.
  // clang-format off
  Eigen::Matrix3d R_tmp = Eigen::AngleAxisd(0.4, Eigen::Vector3d(0, 0, 1)).toRotationMatrix();
  std::cout << "Initial R:\n" << R_tmp << std::endl;
  Sophus::SO3d R(R_tmp);
  // clang-format on

  // iter 10 times.
  for (int iter = 0; iter < 10; iter++) {
    auto R_matrix = R.matrix();
    // Gauss-Newton iteration method.
    Eigen::Matrix3d H = Eigen::Matrix3d::Zero();
    Eigen::Vector3d g = Eigen::Vector3d::Zero();
    for (int i = 0; i < zi.size(); i++) {
      H += J(pi[i], R_matrix) * J(pi[i], R_matrix).transpose();
      g += -J(pi[i], R_matrix) * cost_func(zi[i], pi[i], R_matrix);
    }

    // H * delta_phi = g
    Eigen::Vector3d delta_phi = H.inverse() * g;

    R = Sophus::SO3d::exp(delta_phi) * Sophus::SO3d(R);
  }

  std::cout << "optimized R:\n" << R.matrix() << std::endl;
}