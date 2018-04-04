#include <iostream>
#include <Eigen/Eigen> // <Eigen/Dense> for dense only, <Eigen/Sparse> for sparse only, <Eigen/Eigen> for both

// TUTORIALS                http://eigen.tuxfamily.org/dox/group__DenseMatrixManipulation__chapter.html
// DENSE TUTORIAL:          http://eigen.tuxfamily.org/dox/group__TutorialMatClass.html
// SPARSE TUTORIAL:         http://eigen.tuxfamily.org/dox/group__TutorialSparse.html
// DENSE QUICK REFERENCE:   http://eigen.tuxfamily.org/dox/group__QuickRefPage.html
// SPARSE QUICK REFERENCE:  http://eigen.tuxfamily.org/dox/group__SparseQuickRefPage.html


typedef Eigen::MatrixXd Mat;
typedef Eigen::VectorXd Vec;
typedef Eigen::ArrayXXd Arr; // Note the double X for arrays, unlie the single X for matrices!
//typedef Eigen::ArrayXXi Mask;



// http://www.quantdec.com/misc/MAT8406/Meeting12/SVD.pdf
Vec Ridge(const Mat& X, const Vec& y, double alpha) {
  Eigen::BDCSVD<Mat> svd(X, Eigen::ComputeThinU | Eigen::ComputeThinV);
  Vec singular_values = svd.singularValues();
  Mat diag = (1.0 / (singular_values.array().square() + alpha) * singular_values.array()).matrix().asDiagonal();
  return svd.matrixV() * diag * svd.matrixU().transpose() * y;
}

void FStep(const Mat& Y, const Mat& X, const Arr& omega, Mat& F, double lambda, double epsilon, int max_iter) {
    Mat R(Y.rows(), Y.cols());
    Arr X_row(1, X.cols());
    Arr omega_row(1, omega.cols());

    R = Y - F * X;
    for(int i=0; i < max_iter; ++i) {
        double delta_sq = 0;
        for(int alpha = 0; alpha < F.rows(); ++alpha) {
            for(int beta = 0; beta < F.cols(); ++beta) {
                X_row = X.block(beta, 0, 1, X.cols());
                omega_row = omega.matrix().block(alpha, 0, 1, Y.cols());
                double denom = lambda + (X_row * X_row * omega_row).matrix().sum();
                double numer = (X_row * (R.block(alpha, 0, 1, R.cols()).array() + F(alpha, beta) * X_row) * omega_row).matrix().sum();
                double f = numer/denom;
                R.block(alpha, 0, 1, R.cols()).array() -= (f - F(alpha, beta)) * X_row;
                delta_sq += (f - F(alpha, beta)) * (f - F(alpha, beta));
                F(alpha, beta) = f;
            }
        }
        if(delta_sq < epsilon) {
            return;
        }
    }
}



int main() {
  // Test ridge regression
  Mat M(7,3);
  M << 3,5,7,
       4,6,8,
       4,4,7,
       2,4,6,
       4,7,4,
       7,4,7,
       9,5,3;

  Vec y(7);

  y << 15, 18, 15, 12, 15, 18, 17;

  double alpha = 0.01;

  Vec answer = Ridge(M, y, alpha);
  std::cout << std::endl << answer << std::endl;


  // Test coordinate descent: generate F, X , let Y = FX, pretend you dont know true F, solve Y=FX for F with lambda = 0, should get exactly F

  Arr omega(4, 10); // 1 if known, 0 if missiing
  omega <<
    0,1,1,1,0,0,1,1,1,1,
    1,1,0,1,1,1,0,1,1,1,
    1,1,1,1,1,1,1,1,1,1,
    0,1,1,1,1,1,1,0,1,1;

  Mat X(2,10);
  X <<
    4,8,2,6,0,3,4,1,5,7,
    0,5,8,3,2,5,7,9,4,8;

  Mat F(4, 2);
  F <<
    4,2,
    3,1,
    6,2,
    0,2;

  Mat Y = (F * X).array() * omega;

  Mat F_prime(4, 2);
  F_prime.setRandom();

  std::cout << "Starting point for F" << std::endl << F_prime << std::endl;

  double lambda_f = 0.0;
  double epsilon = 0.0001; // tolerance for coordinate descent
  int max_iter = 100;

  FStep(Y, X, omega, F_prime, lambda_f, epsilon, max_iter);

  std::cout << "True F" << std::endl << F << std::endl;
  std::cout << "Recovered F" << std::endl << F_prime << std::endl;

  return 0;
}
