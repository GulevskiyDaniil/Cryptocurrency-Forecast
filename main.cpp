#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <set>
#include <algorithm>
#include <Eigen/Eigen>
#include <sstream>
#include <unistd.h>
#include <cstdlib>
#include <cstring>

using Mat = Eigen::MatrixXd;
using Vec = Eigen::VectorXd;
using Arr = Eigen::ArrayXXd;
using SpMat = Eigen::SparseMatrix<double>;

Eigen::IOFormat CSVFormat(Eigen::StreamPrecision, Eigen::DontAlignCols, ", ", "\n");

void PrintVector(std::vector<int> v);
void Standardize(Mat& M, Vec* means, Vec* scales);
void Destandardize(const Vec& means, const Vec& scales, Mat& M);


Vec Ridge(const Mat& A, const Vec& y, double alpha);
Vec ChoRidge(const Mat& A, const Vec& y, double alpha);
void FStep(const Mat& Y, const Mat& X, const Arr& Omega, double lambda, double epsilon_F, int max_iter_F, Mat& F);
void ConjugateGradientsInplace(const Mat& A, const Vec& b, int row, double epsilon_cg, Mat& X);
void SparseConjugateGradientsInplace(const SpMat& A, const Vec& b, int row, double epsilon_cg_sparse, Mat& X);
void WStep(const Mat& X, std::vector<int> lags, double lambda_w, double lambda_x, Mat& W);
std::set<int> GetDeltaSet(const std::set<int>& lags, int d);
std::set<int> GetAllNotEmptyDelta(const std::set<int>& lags);
void ModifyG(const Mat& W, const std::set<int>& lags, int W_idx, Mat& G);
void ModifyGSparse(const Mat& W, const std::set<int>& lags, int W_idx, SpMat& G);
void ModifyD(const Mat& W, const std::set<int>& lags, int W_idx, Mat& D);
void ModifyDSparse(const Mat& W, const std::set<int>& lags, int W_idx, SpMat& D);
void ToLaplacian(Mat& G);
void ToLaplacianSparse(SpMat& G);
Mat GetFullW(const Mat& W, const std::set<int> lags_set);
void XStep(const Mat& Y, const Arr& Omega, const Mat& W, const Mat& F, const std::set<int> lags_set,
           double lambda_x, double eta, double epsilon_X, int max_iter_X, Mat& X);
void XStepSparse(const Mat& Y, const Arr& Omega, const Mat& W, const Mat& F, const std::set<int> lags_set,
           double lambda_x, double eta, double epsilon, int max_iter_X, Mat& X);
void Factorize(const Mat& Y, const Arr& Omega, const std::set<int>& lags_set, int rank, double lambda_f, double lambda_w, double lambda_x, double eta,
               double epsilon_X, double epsilon_F, int max_iter_X, int max_iter_F, int max_global_iter, bool sparse, Mat& F, Mat& X, Mat& W);

void MatchByColumn(const Vec& ref, Mat &Y, int col);


template<class T>
void PrintFunction(T object);
void ReadCSV(char* filename, char delimeter, Mat& Y, Arr& Omega);
bool ParseParams(int argc, char* argv[], char** input_file_name, char** output_file_name, char* delimeter, int* rank,
                 int* horizon, int* T, bool* match, std::set<int>* lags_set, bool* sparse, bool* keep_big, double* lambda_x, double* lambda_w, double* lambda_f, double* eta);
Mat GenerateOmega(int rows, int cols, double p_missing);
void TestMatchByColumns();

void Forecast(Mat& Y, const Arr& Omega, const std::set<int>& lags_set, int rank, int horizon, int T, bool match, double lambda_f, double lambda_w, double lambda_x, double eta,
               double epsilon_X, double epsilon_F, int max_iter_X, int max_iter_F, int max_global_iter, bool sparse, bool keep_big, Mat& F, Mat& X, Mat& W);

int Run(int argc, char* argv[]);

/////////////////////////////////////////////////////////////////////


void FitCoeffs() {

    char* input_file_name = "/home/fedor/data/converted.csv";
    Mat Y_orig;
    Arr Omega;
    ReadCSV(input_file_name, ',', Y_orig, Omega);
    std::cout << std::endl << Y_orig.rows() << " " << Y_orig.cols() << std::endl;
    //Mat Y_orig = GetData();
    Mat Y = Y_orig.leftCols(380);

    int n = Y.rows();
    int T = Y.cols();

    std::set<int> lags_set = {1,2,3,4,5,6,7,14,21};

    int k;
    std::cout << std::endl << "input rank: ";
    std::cin >> k;

    int horizon = 1;
    std::cout << std::endl << "input horizon: ";
    std::cin >> horizon;

    double p_missing;
    std::cout << std::endl << "input prob of missing, 0<p<1: ";
    std::cin >> p_missing;
    //Mat Omega = GenerateOmega(Y.rows(), Y.cols()-horizon, p_missing);

    Mat F_low(n, k);
    Mat X_low(k, T-horizon);
    Mat W_low(k, lags_set.size());
    F_low.setRandom();
    X_low.setRandom();
    W_low.setRandom();

    std::vector<double> lambda_x_set = {0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000, 100000};
    std::vector<double> lambda_w_set = {0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000};
    std::vector<double> lambda_f_set = {0.0001, 0.001, 0.01, 0.1, 1};
    std::vector<double> eta_set = {0.0001, 0.001, 0.01, 0.1, 1};

    double epsilon_X = 0.0001;
    double epsilon_F = 0.0001;
    int max_iter_X = 20;
    int max_iter_F = 10;
    int max_global_iter = 10;
    bool sparse = false;
    bool keep_big = false;
    bool match = true;
    double min_rmse = 10000000;
    std::cout << std::endl;

    for(auto lambda_x: lambda_x_set) {
        for(auto lambda_w: lambda_w_set) {
            for(auto lambda_f: lambda_f_set) {
                for (auto eta: eta_set) {
                    std::cout << "." << std::flush;
                    Mat Y = Y_orig.leftCols(380);
                    Vec means, scales;
                    Standardize(Y, &means, &scales);
                    Mat Y_block = Y.leftCols(Y.cols() - horizon);
                    Arr Omega_block = Omega.leftCols(Y.cols() - horizon);
                    Forecast(Y_block, Omega_block, lags_set, k, horizon, T, match, lambda_f, lambda_w, lambda_x, eta, epsilon_X, epsilon_F,
                             max_iter_X, max_iter_F, max_global_iter, sparse, keep_big, F_low, X_low, W_low);
                    //std::cout << std::endl << Y_block.cols()  << " " << Y.cols() << std::endl;
                    double rmse = ((Y_block.rightCols(horizon) - Y.rightCols(horizon)).array() * (Y_block.rightCols(horizon) - Y.rightCols(horizon)).array()).mean();
                    if (rmse<min_rmse) {
                        min_rmse = rmse;
                        std::cout << std::endl << "lambda_x: " << lambda_x << ";\t lambda_w: " << lambda_w << ";\t lambda_f: " << lambda_f << "; eta: " << eta << ";\t rmse:" << rmse;
                    }
                }
            }
        }
    }
    std::cout << std::endl;
    //rank: 6 horizon 20 p_missing 0 lambda_x: 10000;  lambda_w: 1000;   lambda_f: 0.001; eta: 0.0001;   rmse:0.421484.
    //                               lambda_x: 100000;   lambda_w: 1000;   lambda_f: 0.0001; eta: 0.01;  rmse:0.396559.
}



///////////////////////////////////////////////////////////////////////////

int main(int argc, char* argv[] ) {
    // lambda_x: 100;	 lambda_w: 1;	 lambda_f: 10000; eta: 0.0001
    //FitCoeffs();
    return Run(argc, argv);;
}

///////////////////////////////////////////////////////////////////////////

int Run(int argc, char* argv[]) {
    if (argc < 10) {
        std::cerr << std::endl << "Usage: " << argv[0] <<
                     " -i input_file -o prediction_file -d delimeter -k rank -h horizon -s (if you want to use sparse algorithm) -x lambda_x -w lambda_w -f lambda_f -e eta" << std::endl;
        return 1;
    }

    char* input_file_name ;
    char* output_file_name;

    char delimeter = ',';
    int rank, horizon, T = -1;
    bool sparse = false;
    bool keep_big = false;
    bool match = false;
    double lambda_x=1, lambda_w=1, lambda_f=1, eta=1;

    std::set<int> lags_set;
    ParseParams(argc, argv, &input_file_name, &output_file_name, &delimeter, &rank, &horizon, &T, &match, &lags_set, &sparse, &keep_big, &lambda_x, &lambda_w, &lambda_f, &eta);
    Mat Y;
    Arr Omega;
    ReadCSV(input_file_name, ',', Y, Omega);
    //std::cout << std::endl << Y.rows() << " " << Y.cols() << std::endl;
    //return 0;

    int n = Y.rows();
    if (T == -1) T = Y.cols();

    double epsilon_X = 0.0001;
    double epsilon_F = 0.0001;
    int max_iter_X = 10;
    int max_iter_F = 10;
    int max_global_iter = 20;

    //Vec means, scales; // DELETE ME
    //Standardize(Y, &means, &scales); // DELETE ME

    Mat Y_pred = Y.leftCols(T);
    Arr Omega_part = Omega.leftCols(T);

    Mat F, X;
    Mat W(rank, lags_set.size());
    Forecast(Y_pred, Omega_part, lags_set, rank, horizon, T, match, lambda_f, lambda_w, lambda_x, eta, epsilon_X, epsilon_F,
             max_iter_X, max_iter_F, max_global_iter, sparse, keep_big, F, X, W);



    std::ofstream file("bob.csv");
    file << Y.format(CSVFormat);
    file.close();

    std::ofstream output_file(output_file_name);
    output_file << Y_pred.format(CSVFormat);

    return 0;
}


void Forecast(Mat& Y, const Arr& Omega, const std::set<int>& lags_set, int rank, int horizon, int T, bool match, double lambda_f, double lambda_w, double lambda_x, double eta,
               double epsilon_X, double epsilon_F, int max_iter_X, int max_iter_F, int max_global_iter, bool sparse, bool keep_big, Mat& F, Mat& X, Mat& W) {

    int n = Y.rows();

    Vec means, scales;
    Standardize(Y, &means, &scales);

    Y = Y.array() * Omega;

    Vec reference_column = Y.col(Y.cols() - 1);

    Factorize(Y, Omega, lags_set, rank, lambda_f, lambda_w, lambda_x, eta, epsilon_X, epsilon_F, max_iter_X, max_iter_F, max_global_iter, sparse, F, X, W);

    X.conservativeResize(rank, T + horizon);

    std::vector<int> lags_vec(lags_set.begin(), lags_set.end());
    std::sort(lags_vec.begin(), lags_vec.end());

    for (int row=0; row<rank; ++row) {
        for(int t=T; t<T+horizon; ++t) {
            double value = 0;
            for (int lag_idx=0; lag_idx<lags_vec.size(); ++lag_idx) {
                value += X(row, t - lags_vec[lag_idx]) * W(row, lag_idx);
            }
            X(row, t) = value;
        }
    }

    if(keep_big) {
        Y = F * X;
        if(match) MatchByColumn(reference_column, Y, T-1);
    } else {
        Mat X_pred = X.rightCols(horizon+1);
        Mat Y_pred = F * X_pred;
        if(match) MatchByColumn(reference_column, Y_pred, 0);
        Y = Y_pred.rightCols(horizon);
    }
    Destandardize(means, scales, Y); //UNCOMMENT ME
}


void Factorize(const Mat& Y, const Arr& Omega, const std::set<int>& lags_set, int rank, double lambda_f, double lambda_w, double lambda_x, double eta,
               double epsilon_X, double epsilon_F, int max_iter_X, int max_iter_F, int max_global_iter, bool sparse, Mat& F, Mat& X, Mat& W) {
    /*
    Given:
        Y:         an n x T matrix, where each row is a time series and each column is a time tick. If the value [i,j] is missing, the corresponding Y[i,j] must be zero!
        Omega:     an n x T mask matrix , which has 1 in position i,j if Y[i,j] is known, and 0 if it is unknown.
        lags:      a set of integers, corresponding to the lags that the autoregressive model includes.
        k:         the rank of decomposition
        lambda_f:  regularization coefficient for Frobenius norm of F
        lambda_w:  regularization coefficient for Frobenius norm of W
        lambda_x:  regularization coefficient for the temporal regularizer T_{AR}(X)
        eta:       regularization coefficient for Frobenius norm of X inside T_{AR}(X)

        The problem being solved is:
        ||P_{Omega}(Y - FX)||^2  + lambda_f ||F|| + lambda_w ||W|| + lambda_x (\sum_{r=1}^k T_{AR}(X_k)) -> min_{F, X, W}
    */

   std::vector<int> lags_vec(lags_set.begin(), lags_set.end());

    // Create first approximations for F and X using the SVD decomposition

   Eigen::BDCSVD<Mat> svd(Y, Eigen::ComputeThinU | Eigen::ComputeThinV);

   F = svd.matrixU().leftCols(rank) * svd.singularValues().head(rank).asDiagonal();
   X = svd.matrixV().leftCols(rank).transpose();

    for(int iter=0; iter<max_global_iter; ++iter){
        std::cout << "Iteration " << iter + 1 << std::endl;
        Mat X_prev = X;
        WStep(X, lags_vec, lambda_w, lambda_x, W);

        if (sparse) {
           XStepSparse(Y, Omega, W, F, lags_set, lambda_x, eta, epsilon_X, max_iter_X, X);
        } else {
           XStep(Y, Omega, W, F, lags_set, lambda_x, eta, epsilon_X, max_iter_X, X);
        }

        FStep(Y, X, Omega, lambda_f, epsilon_F, max_iter_F, F);

        double diff_X_norm = (X - X_prev).norm();
        //std::cout << "\t mean X diff sq " << diff_X_norm / (X.rows() * X.cols());
        if ((diff_X_norm) < epsilon_X * X.rows() * X.cols()) {
           return;
        }
    }
}



void PrintVector(std::vector<int> v) {
    std::cout << std::endl;
    for (std::vector<int>::const_iterator i = v.begin(); i != v.end(); ++i)
        std::cout << *i << ' ';
    std::cout << std::endl;
}


void Standardize(Mat& M, Vec* means, Vec* scales) {
    *means = M.rowwise().mean();
    M = M.colwise() - *means;
    *scales = (M.array() * M.array()).rowwise().mean().sqrt();
    M = M.array().colwise() / scales->array();
}

void Destandardize(const Vec& means, const Vec& scales, Mat& M) {
    M = M.array().colwise() * scales.array();
    M = M.colwise() + means;
}

void MatchByColumn(const Vec& ref, Mat &Y, int col) {
    Vec col_values = Y.col(col);
    Y  = (Y.colwise() + ref).colwise() - col_values;
    for(int row=0; row< Y.rows(); ++row) {
        //std::cout << std::endl << col << "\t" << Y(row,col) << "\t" << ref(row) << std::endl;
    }
}


// http://www.quantdec.com/misc/MAT8406/Meeting12/SVD.pdf page 3
Vec Ridge(const Mat& A, const Vec& y, double alpha) {
    /*
    Solves ridge regression problem with object-feature matrix A, target y and l2 regularization coeff alpha
    */
    Eigen::BDCSVD<Mat> svd(A, Eigen::ComputeThinU | Eigen::ComputeThinV);
    Vec singular_values = svd.singularValues();
    Mat diag = (1.0 / (singular_values.array().square() + alpha) * singular_values.array()).matrix().asDiagonal();
    return svd.matrixV() * diag * svd.matrixU().transpose() * y;
}

Vec ChoRidge(const Mat& A, const Vec& y, double alpha) {
    Eigen::LDLT<Mat> ldlt;
    Mat I(A.cols(), A.cols());
    I.setIdentity();
    ldlt.compute(A.transpose() * A + alpha * I);
    Eigen::VectorXd w_r = ldlt.solve(A.transpose() * y);
    return w_r;
}


// http://www.cs.utexas.edu/~cjhsieh/icdm-pmf.pdf Section IIIA
void FStep(const Mat& Y, const Mat& X, const Arr& Omega, double lambda, double epsilon_F, int max_iter_F, Mat& F) {
    /*
    Solves problem ||P_omega(Y - F * X)|| + lambda * ||F|| --F--> min using coordinate descent algorithm
    omega is an array with same shape as Y with ones where Y_ij is observed and zeros where Y_ij is missing.
    P_omega is the projector in the space of matrices onto the subspace generated by observed indices.
    The iterative process is stopped if the sum of squares of changes of elements of F is less than epsilon.
    The problem is solved inplace, so F is overwritten with the answer.
    */
    Mat R = Y - F * X;
    for(int i=0; i < max_iter_F; ++i) {
        double delta_sq = 0;
        for(int row = 0; row < F.rows(); ++row) {
            for(int col = 0; col < F.cols(); ++col) {
                double denom = lambda + (X.row(col).array() * X.row(col).array() * Omega.row(row).array()).matrix().sum();
                double numer = (X.row(col).array() * (R.row(row).array() + F(row, col) * X.row(col).array()) * Omega.row(row).array()).matrix().sum();
                double f = numer/denom;
                R.row(row).array() -= (f - F(row, col)) * X.row(col).array();
                delta_sq += (f - F(row, col)) * (f - F(row, col));
                F(row, col) = f;
            }
        }
        if(delta_sq < epsilon_F * F.rows() * F.cols()) {
            return;
        }
    }
}

// https://en.wikipedia.org/wiki/Conjugate_gradient_method#The_resulting_algorithm
void ConjugateGradientsInplace(const Mat& A, const Vec& b, int row, double epsilon_cg, Mat& X){
    /*
    Solves Ax = b for symmetric positive definite A, where x = X.row(row)
    */
    Vec r0 = b - A * X.row(row).transpose();
    Vec r1 = r0;
    Vec p = r0;
    for (int k=0; k<A.rows(); ++k) {
        double alpha = r0.dot(r0) / (p.transpose() *  A * p);
        X.row(row) += (alpha * p).transpose();
        r1 = r0 - alpha * A * p;
        if (r1.norm() < epsilon_cg * X.cols()) {
            return;
        }
        double beta = r1.dot(r1) / r0.dot(r0);
        p = r1 + beta * p;
        r0 = r1;
    }
    return;
}


// https://en.wikipedia.org/wiki/Conjugate_gradient_method#The_resulting_algorithm
void SparseConjugateGradientsInplace(const SpMat& A, const Vec& b, int row, double epsilon_cg_sparse, Mat& X){
    /*
    Solves AX=b for sparse symmetric positive definite A.
    Only the upper half of A should be filled, the lower part will be ignored.
    */
    Vec r0 = b - A.selfadjointView<Eigen::Upper>() * X.row(row).transpose();
    Vec r1 = r0;
    Vec p = r0;
    for (int k=0; k<A.rows(); ++k) {
        double alpha = r0.dot(r0) / (p.transpose() *  A.selfadjointView<Eigen::Upper>() * p);
        X.row(row).transpose() += alpha * p;
        r1 = r0 - A.selfadjointView<Eigen::Upper>() * p * alpha;
        if (r1.norm() < epsilon_cg_sparse * X.cols()) {
            return;
        }
        double beta = r1.dot(r1) / r0.dot(r0);
        p = r1 + beta * p;
        r0 = r1;
    }
    return;
}


void WStep(const Mat& X, std::vector<int> lags, double lambda_w, double lambda_x, Mat& W) {
    /*
    Solves for autoregression weights with given matrix X. All this function does is initialize
    matrices and vectors for ridge regression
    */
    std::sort(lags.begin(), lags.end());
    int T = X.cols();
    int k = X.rows();
    int l = lags.size();
    int L = lags.back();
    double alpha = 0.5 * lambda_w / lambda_x;

    for (int row =  0; row < k; ++row) {
        Mat M(T - L, l);
        Vec y(T - L);
        for (int i=L; i<T; ++i) {
            y(i - L) = X(row, i);
            for(int j=0; j<l; ++j) {
                M(i - L, l - 1 -j) = X(row, i - lags[j]);
            }
        }
        W.row(row) = ChoRidge(M, y, alpha).transpose();
    }
}


std::set<int> GetDeltaSet(const std::set<int>& lags, int d){
    std::set<int> deltaset;
    int lag = 0;
    if (lag - d == 0) {
        deltaset.insert(lag);
    }
    for (int lag : lags) {
        if (lag - d == 0) {
            deltaset.insert(lag);
        } else if (lags.find(lag - d) != lags.end()) {
            deltaset.insert(lag);
        }
    }
    return deltaset;
}


std::set<int> GetAllNotEmptyDelta(const std::set<int>& lags){
    std::set<int> deltas;
    int delta = 0;
    deltas.insert(delta);

    for (int lag1 : lags) {
        int lag2 = 0; {
            delta = lag1 - lag2;
            if (delta >= 0) {
                deltas.insert(delta);
            }
        }
    }

    for (int lag1 : lags) {
        for (int lag2 : lags) {
            delta = lag1 - lag2;
            if (delta >= 0) {
                deltas.insert(delta);
            }
        }
    }
    return deltas;
}


void ModifyG(const Mat& W, const std::set<int>& lags, int W_idx, Mat& G) {
    double w_0 = -1;
    int T = G.cols();
    int L = *(std::max_element(lags.begin(), lags.end()));
    int m = 1 + L;
    std::set<int> deltas = GetAllNotEmptyDelta(lags);

    for (int idx=0; idx<G.rows(); ++idx) {
        int t = idx + 1;
        for (int d : deltas) {
            if ((d + idx) < G.cols()) {
                //ToDo save deltaset
                std::set<int> deltaset = GetDeltaSet(lags, d);
                double value = 0.0;
                for (int l : deltaset) {
                    if ((m <= (t + l)) && ((t + l) <= T)) {
                        if ((l - d == 0) && (l == 0)) {
                            value += - w_0 * w_0;
                        } else if (l - d == 0) {
                            value += - W(W_idx,l - 1) * w_0;
                        } else if (l == 0) {
                            value += - w_0 * W(W_idx,l - d - 1);
                        } else {
                            value += - W(W_idx,l - 1) * W(W_idx,l - d - 1);
                        }
                    }
                }
                G(t - 1, t + d - 1) = value;
                G(t + d - 1, t - 1) = value;
            }
        }
    }
}


void ModifyGSparse(const Mat& W, const std::set<int>& lags, int W_idx, SpMat& G) {
    double w_0 = -1;
    int T = G.cols();
    int L = *(std::max_element(lags.begin(), lags.end()));
    int m = 1 + L;
    std::set<int> deltas = GetAllNotEmptyDelta(lags);

    for (int idx=0; idx<G.rows(); ++idx) {
        int t = idx + 1;
        for (int d : deltas) {
            if ((d + idx) < G.cols()) {
                //ToDo save deltaset
                std::set<int> deltaset = GetDeltaSet(lags, d);
                double value = 0.0;
                for (int l : deltaset) {
                    if ((m <= (t + l)) && ((t + l) <= T)) {
                        if ((l - d == 0) && (l == 0)) {
                            value += - w_0 * w_0;
                        } else if (l - d == 0) {
                            value += - W(W_idx,l - 1) * w_0;
                        } else if (l == 0) {
                            value += - w_0 * W(W_idx,l - d - 1);
                        } else {
                            value += - W(W_idx,l - 1) * W(W_idx,l - d - 1);
                        }
                    }
                }
                G.insert(t - 1, t + d - 1) = value;
            }
        }
    }
}


void ModifyD(const Mat& W, const std::set<int>& lags, int W_idx, Mat& D) {
    double w_0 = -1;
    int T = D.cols();
    int L = *(std::max_element(lags.begin(), lags.end()));
    int m = 1 + L;
    double w_sum = 0.0;
    std::set<int> lags_hat = lags;
    lags_hat.insert(0);

    w_sum += w_0;
    for (int l : lags) {
        w_sum += W(W_idx,l - 1);
    }

    for (int idx=0; idx<D.rows(); ++idx) {
        int t = idx + 1;
        double value = 0.0;
        for (int l : lags_hat) {
            if ((m <= (t + l)) && ((t + l) <= T)) {
                if (l == 0) {
                    value += w_sum * w_0;
                } else {
                    value += w_sum * W(W_idx,l - 1);
                }
            }
            D(t - 1, t - 1) = value;
        }
    }
}


void ModifyDSparse(const Mat& W, const std::set<int>& lags, int W_idx, SpMat& D) {
    double w_0 = -1;
    int T = D.cols();
    int L = *(std::max_element(lags.begin(), lags.end()));
    int m = 1 + L;
    double w_sum = 0.0;
    std::set<int> lags_hat = lags;
    lags_hat.insert(0);

    w_sum += w_0;
    for (int l : lags) {
        w_sum += W(W_idx,l - 1);
    }

    for (int idx=0; idx<D.rows(); ++idx) {
        int t = idx + 1;
        double value = 0.0;
        for (int l : lags_hat) {
            if ((m <= (t + l)) && ((t + l) <= T)) {
                if (l == 0) {
                    value += w_sum * w_0;
                } else {
                    value += w_sum * W(W_idx,l - 1);
                }
            }
        }
        D.insert(t - 1, t - 1) = value;
    }
}


void ToLaplacian(Mat& G) {
    G *= -1;
    G -= G.colwise().sum().asDiagonal();
}


void ToLaplacianSparse(SpMat& G) {
    /*
       Transforms adjacency matrix G to Laplace matrix inplace.
    */
    int T = G.rows();
    Vec unit_vector(T);
    unit_vector.setOnes();

    SpMat degree_matrix(T,T);
    degree_matrix.setIdentity(); //a nifty way to allocate diagonal elements. Otherwise the line below doesnt work
    degree_matrix.diagonal() = G.selfadjointView<Eigen::Upper>() * unit_vector;
    G *= -1;
    G += degree_matrix;
}


Mat GetFullW(const Mat& W, const std::set<int> lags_set) {
    std::vector<int> lags_vec(lags_set.begin(), lags_set.end());
    std::sort(lags_vec.begin(), lags_vec.end());

    Mat W_full(W.rows(), lags_vec.back());
    W_full.setZero();
    for(int col=0; col<W.cols(); ++col) {
        W_full.col(lags_vec[col] - 1) = W.col(col);
    }
    return W_full;
}


void XStep(const Mat& Y, const Arr& Omega, const Mat& W, const Mat& F, const std::set<int> lags_set,
           double lambda_x, double eta, double epsilon_X, int max_iter_X, Mat& X) {
    /*
        Solve for X with fixed W and F.
    */

    Mat W_full = GetFullW(W, lags_set);

    int T = Y.cols();

    Mat I(T, T);
    I.setIdentity();

    for(int iter=0; iter<max_iter_X; ++iter) {
        Mat X_prev = X;
        for(int row=0; row<X.rows(); ++row) {
            Mat G(T, T);
            Mat D(T, T);
            G.setZero();
            D.setZero();

            ModifyG(W_full, lags_set, row, G);
            ModifyD(W_full, lags_set, row, D);

            G.diagonal().setZero();
            ToLaplacian(G);

            G *= 0.5;
            G += 0.5 * eta * I + 0.5 * D;
            G *= lambda_x;

            Mat Y_tilde(Y.rows(), Y.cols());
            Y_tilde = Y - F * X + F.col(row) * X.row(row);

            Vec b(T);

            for (int j=0; j<T; ++j) {
                b(j) = (F.col(row).array() * F.col(row).array() * Omega.col(j)).sum();
            }

            G.diagonal() += b;

            Vec lhs(T);
            lhs = Y.transpose() * F.col(row);
            ConjugateGradientsInplace(G, lhs, row, 0.0001, X);
        }
        if((X - X_prev).norm() < epsilon_X * X.rows() * X.cols()) {
            break;
        }
    }
}


void XStepSparse(const Mat& Y, const Arr& Omega, const Mat& W, const Mat& F, const std::set<int> lags_set,
           double lambda_x, double eta, double epsilon, int max_iter_X, Mat& X) {
    /*
        Solve for X with fixed W and F.
    */

    Mat W_full = GetFullW(W, lags_set);

    int T = Y.cols();
    // Now iterate over rows of X and W

    SpMat I(T, T);
    I.setIdentity();

    for(int iter=0; iter<max_iter_X; ++iter) {
        Mat X_prev = X;
        for(int row=0; row<X.rows(); ++row) {
            SpMat G(T, T);
            G.reserve(T * lags_set.size());
            SpMat D(T, T);
            D.reserve(T);

            ModifyGSparse(W_full, lags_set, row, G);
            ModifyDSparse(W_full, lags_set, row, D);

            G.diagonal().setZero();
            ToLaplacianSparse(G);

            G *= 0.5;
            G += 0.5 * eta * I + 0.5 * D;
            G *= lambda_x;

            Mat Y_tilde(Y.rows(), Y.cols());
            Y_tilde = Y - F * X + F.col(row) * X.row(row);

            Vec b(T);

            for (int j=0; j<T; ++j) {
                b(j) = (F.col(row).array() * F.col(row).array() * Omega.col(j)).sum();
            }

            G.diagonal() += b;

            Vec lhs(T);
            lhs = Y.transpose() * F.col(row);
            SparseConjugateGradientsInplace(G, lhs, row, 0.0001, X);
        }
        if((X - X_prev).norm() < epsilon * X.rows() * X.cols()) {
            break;
        }
    }
}


template<class T>
void PrintFunction(T object) {
    for(auto it = object.begin(); it != object.end(); it++) {
        std::cout << *it << ", ";
    }
    std::cout << std::endl;
}


void ReadCSV(char* filename, char delimeter, Mat& Y, Arr& Omega) {

    Y.conservativeResize(0, 0);
    Omega.conservativeResize(0,0);
    std::ifstream input_stream;
    input_stream.open(filename);

    std::string line;
    int row = 0;
    while (std::getline(input_stream, line)) {
    //std::replace(line.begin(), line.end(), delimeter, ' '); // fixme
    Y.conservativeResize(row + 1, Y.cols());
    Omega.conservativeResize(row + 1, Y.cols());

    std::istringstream iss(line);
    //std::cout << std::endl << line;
    std::string value;
    int col = 0;
    while (std::getline(iss, value, delimeter)) {
        if (Y.cols() < col + 1) {
            Y.conservativeResize(Y.rows(), col + 1);
            Omega.conservativeResize(Y.rows(), col + 1);
        }
        try {
            Y(row, col) = std::stod(value);
            Omega(row, col) = 1;
        } catch(std::invalid_argument) {
            Y(row, col) = 0;
            Omega(row, col) = 0;
        }
        col +=1;
    }
    row += 1;
    }
    input_stream.close();
}


bool ParseParams(int argc, char* argv[], char** input_file_name, char** output_file_name, char* delimeter, int* rank, int* horizon, int* T, bool* match,
                 std::set<int>* lags_set, bool* sparse, bool* keep_big, double* lambda_x, double* lambda_w, double* lambda_f, double* eta) {
    char* lags;
    int c;
    while( ( c = getopt(argc, argv, "i:o:d:k:l:z:T:smbx:w:f:e:") ) != -1 ) {
        switch(c) {
            case 'i':
                *input_file_name = optarg;
                break;
            case 'o':
                *output_file_name = optarg;
                break;
            case 'd':
                if(optarg) *delimeter = optarg[0];
                break;
            case 'k':
                *rank = atoi(optarg);
                break;
            case 'l':
                lags = optarg;
                break;
            case 'z':
                if(optarg) *horizon = atoi(optarg);
                break;
            case 'T':
                if(optarg) *T = atoi(optarg);
                break;
             case 'm':
                *match = true;
                break;
            case 's':
                *sparse = true;
                break;
            case 'b':
                *keep_big = true;
                break;
            case 'x':
                if(optarg) *lambda_x = atof(optarg);
                break;
            case 'w':
              if(optarg) *lambda_w = atof(optarg);
                break;
            case 'f':
              if(optarg) *lambda_f = atof(optarg);
               break;
            case 'e':
              if(optarg) *eta = atof(optarg);
                break;
        }
    }
    std::string lags_string(lags);
    std::istringstream iss(lags_string);
    std::string lag;
    while (std::getline(iss, lag, ',')) {
        lags_set->insert(std::stoi(lag));
    }
    return true;
}


Mat GenerateOmega(int rows, int cols, double p_missing) {
    Arr Omega(rows, cols);
    Omega.setOnes();
    for (int i=0; i<Omega.rows(); ++i) {
        for(int j=0; j<Omega.cols(); ++j) {
            int random_number = std::rand();
            if (random_number < RAND_MAX * p_missing) {
                Omega(i,j) = 0;
            }
        }
    }
    return Omega;

}


