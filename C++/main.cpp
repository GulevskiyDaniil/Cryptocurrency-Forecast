#include <iostream>
#include <Eigen/Eigen> // <Eigen/Dense> for dense only, <Eigen/Sparse> for sparse only, <Eigen/Eigen> for both

// TUTORIALS                http://eigen.tuxfamily.org/dox/group__DenseMatrixManipulation__chapter.html
// DENSE TUTORIAL:          http://eigen.tuxfamily.org/dox/group__TutorialMatClass.html
// SPARSE TUTORIAL:         http://eigen.tuxfamily.org/dox/group__TutorialSparse.html
// DENSE QUICK REFERENCE:   http://eigen.tuxfamily.org/dox/group__QuickRefPage.html
// SPARSE QUICK REFERENCE:  http://eigen.tuxfamily.org/dox/group__SparseQuickRefPage.html


typedef Eigen::MatrixXd Mat;
typedef Eigen::VectorXd Vec;
typedef Eigen::ArrayXXd Arr; // Note the double X for arrays, unlike the single X for matrices!

// http://www.quantdec.com/misc/MAT8406/Meeting12/SVD.pdf
Vec Ridge(const Mat& X, const Vec& y, double alpha) {
    Eigen::BDCSVD<Mat> svd(X, Eigen::ComputeThinU | Eigen::ComputeThinV);
    Vec singular_values = svd.singularValues();
    Mat diag = (1.0 / (singular_values.array().square() + alpha) * singular_values.array()).matrix().asDiagonal();
    return svd.matrixV() * diag * svd.matrixU().transpose() * y;
}

void FStep(const Mat& Y, const Mat& X, const Arr& omega, Mat& F, double lambda, double epsilon, int max_iter) {
    Mat R = Y - F * X;
    for(int i=0; i < max_iter; ++i) {
        double delta_sq = 0;
        for(int row = 0; row < F.rows(); ++row) {
            for(int col = 0; col < F.cols(); ++col) {
                double denom = lambda + (X.row(col).array() * X.row(col).array() * omega.row(row).array()).matrix().sum();
                double numer = (X.row(col).array() * (R.row(row).array() + F(row, col) * X.row(col).array()) * omega.row(row).array()).matrix().sum();
                double f = numer/denom;
                R.block(row, 0, 1, R.cols()).array() -= (f - F(row, col)) * X.row(col).array();
                delta_sq += (f - F(row, col)) * (f - F(row, col));
                F(row, col) = f;
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
    omega << 0,1,1,1,0,0,1,1,1,1,
             1,1,0,1,1,1,0,1,1,1,
             1,1,1,1,1,1,1,1,1,1,
             0,1,1,1,1,1,1,0,1,1;

    Mat X(2,10);
    X << 4,8,2,6,0,3,4,1,5,7,
         0,5,8,3,2,5,7,9,4,8;

    Mat F(4, 2);
    F << 4,2,
         3,1,
         6,2,
         0,2;

    Mat Y = (F * X).array() * omega;

    Mat F_prime(4, 2);
    F_prime.setRandom();

    std::cout << "Starting point for F" << std::endl << F_prime << std::endl;

    double lambda_f = 0.0;
    double epsilon = 0.00000001; // tolerance for coordinate descent
    int max_iter = 1000;

    FStep(Y, X, omega, F_prime, lambda_f, epsilon, max_iter);

    std::cout << "True F" << std::endl << F << std::endl;
    std::cout << "Recovered F" << std::endl << F_prime << std::endl;

    return 0;
}
