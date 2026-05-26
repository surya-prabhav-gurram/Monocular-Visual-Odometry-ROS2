#include <gtsam/nonlinear/NonlinearFactorGraph.h>
#include <gtsam/nonlinear/Values.h>
#include <gtsam/nonlinear/LevenbergMarquardtOptimizer.h>
#include <gtsam/geometry/Rot3.h>
#include <gtsam/inference/Symbol.h>
#include <gtsam/linear/NoiseModel.h>     // ✅ correct include for Jazzy build
#include <gtsam/base/Vector.h>
#include <gtsam/base/numericalDerivative.h>
#include <iostream>
#include <vector>

using namespace gtsam;

// Flatten 3x3 rotation to 9x1 vector (row-major)
static Vector9 flattenRowMajor(const Rot3& R) {
  Matrix3 M = R.matrix();
  Vector9 v;
  int k = 0;
  for (int r = 0; r < 3; ++r)
    for (int c = 0; c < 3; ++c)
      v(k++) = M(r,c);
  return v;
}

// Factor for Frobenius-norm rotation averaging
class FrobeniusRotationFactor : public NoiseModelFactor1<Rot3> {
  Rot3 Ri_;

  Vector errorVector(const Rot3& R) const {
    return flattenRowMajor(R) - flattenRowMajor(Ri_);
  }

public:
  FrobeniusRotationFactor(Key key, const Rot3& Ri, const SharedNoiseModel& model)
    : NoiseModelFactor1<Rot3>(model, key), Ri_(Ri) {}

  Vector evaluateError(const Rot3& R,
                       boost::optional<Matrix&> H = boost::none) const override {
    if (H) {
      auto f = [this](const Rot3& X){ return this->errorVector(X); };
      *H = numericalDerivative11<Vector, Rot3>(f, R);
    }
    return errorVector(R);
  }
};

int main() {
  NonlinearFactorGraph graph;
  Values initial;

  // Example rotation measurements (simulate noisy data)
  std::vector<Rot3> Ri{
    Rot3::RzRyRx( 0.10,  0.00,  0.00),
    Rot3::RzRyRx( 0.00,  0.18,  0.02),
    Rot3::RzRyRx(-0.08,  0.12,  0.07),
    Rot3::RzRyRx( 0.05, -0.05,  0.01),
    Rot3::RzRyRx( 0.00,  0.00,  0.10)
  };

  Symbol Rkey('R', 0);
  auto noise = gtsam::noiseModel::Diagonal::Sigmas(
    (Vector(9) << Vector9::Constant(0.05)).finished());

  for (const auto& Rm : Ri)
    graph.add(boost::make_shared<FrobeniusRotationFactor>(Rkey, Rm, noise));

  initial.insert(Rkey, Rot3::Identity());

  LevenbergMarquardtParams params;
  params.setVerbosity("ERROR");
  params.setMaxIterations(50);
  LevenbergMarquardtOptimizer optimizer(graph, initial, params);
  Values result = optimizer.optimize();

  std::cout << "Initial Rotation:\n"   << initial.at<Rot3>(Rkey).matrix() << "\n\n";
  std::cout << "Optimized Rotation*:\n" << result.at<Rot3>(Rkey).matrix()  << "\n\n";

  // Closed-form check: projection of sum(Ri)
  Matrix3 N = Matrix3::Zero();
  for (const auto& Rm : Ri) N += Rm.matrix();
  Eigen::JacobiSVD<Matrix3> svd(N, Eigen::ComputeFullU | Eigen::ComputeFullV);
  Matrix3 Rproj = svd.matrixU() * svd.matrixV().transpose();
  if (Rproj.determinant() < 0) {
    Matrix3 U = svd.matrixU();
    U.col(2) *= -1.0;
    Rproj = U * svd.matrixV().transpose();
  }
  std::cout << "Closed-form average (Proj(sum Ri)):\n" << Rproj << "\n";
  return 0;
}
