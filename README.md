# TRMF
This is a C++ implementation of the basic Temporal Regularized Matrix Factorization (TRMF) algorithm proposed by Hsiang-Fu Yu, Nikhil Rao and Inderjit S. Dhillon in  [their 2016 NIPS paper](https://papers.nips.cc/paper/6160-temporal-regularized-matrix-factorization-for-high-dimensional-time-series-prediction). The implementation uses the Eigen 3 library for dense andsparse linear algebra routines.

This program was created by Daniil Gulevskiy and Fedor Indukaev as aa course project for the Large Scale Machine Learning course at [Yandex school of data analysis](https://yandexdataschool.com/).

## About the TRMF algorithm
The TRMF algorihm seeks to solve the problem of multiple time series forecasting, as this is known to be a hard problem for conventional forecasting algorithms such as ARIMA and DLM. Simple versions of these algorithms only work with each time series separately, thus disregarding the correlations among them. Multidimensional DLM model does exist, but scales very poorly with the dimension. Also these conventional models cannot handle missing values, so heuristics should be used to address missing data.

Suppose we have `n` time series spanned over the same time frame of T time ticks, all organized into a matrix `Y` with `n` rows and `T` columns. The general idea of TRMF is to to find a decomposition `Y ≃ FX` of low rank `k` (where `F` is `n x k` and `X` is `k x T`) and at the same time to infer a time series forecasting model on the matrix `X`. The rows of the matrix `X` can be thought of as the basis time series of the `k`-dimensional latent subspace in the `n`-dimensional space of the original time series.

The forecasting is then done in the `X`-space, and the predicted values are sent back to the `Y`-space with the matrix `F`.

This also naturally allows for missing data, as matrix factorizations are routinely used to the very purpose of missing value imputation, for example in recommender systems.

The 'basic' TRMF algorithm is when the time series model in the X space is just a simple autoregressive model, separate for each row of `X`. The `k` models share the lag set, but have different weights. This allows to avoid the scalability problem, while still accounting for possible correlations in data, as they are captured in the `F`-matrix. 

The decomposition and time series model are found by solving the following optimization problem:

We are given 
* The matrix `Y`
* The mask matrix Omega of the same shape as `Y`, which has zeros where the values of `Y` are missing and ones elsewhere
* a lag set LS, denote `L = max(LS)` 
* set of regularization coefficients `lambda_F`, `lambda_W`, `lambda_X`, `eta`. 

Optimize

![main formula](https://i.imgur.com/OIfOZDE.png)
where 
![another formula](https://i.imgur.com/CE8q6cY.png)

`W` denotes the `k  x |LS|` matrix of autoregression weights.  The norm operator denotes the Frobenius norm.

The optimization is done by alternately optimizing for `X`, `W` and `F`. For each of the three steps different algorithms are used

## Installing

Just clone the repository and run `make` in the directory with the source files. 


## Usage
`./tmrf -i input_file -o prediction_file -d delimeter -k rank -z horizon -l lags_set -s (if you want to use sparse algorithm) -m (if you want to match predicted series to last known values of true series) -b (if you want the program to to output the big recovered matrix rather than just the predicted values) -x lambda_x -w lambda_w -f lambda_f -e eta`

* `-i input_file` - location of the input file. The file should be in CSV format with one line corresponding to one time series. Missing values can be marked by any non-numeric character, like `#` or 'n';
* `-o prediction_file` - where to write the predictions;
* `-d delimeter` - a character separatimg the values in the input file. For example `-d ,`;
* `-k rank` - rank of the factorization;
* `-z horizon` - how many time ticks you want to predict;
* `-m` - if you want to shift the predicted part to series, so that they match the terminal values in data. Recommended for prediction.
* `-l lags_set` - the lags you want to include into the autoregressive model, separated by comma (no spaces in between). For example, `-l 1,2,7`
* `-s` - with this flag, sparse matrices are used in the `X` - step. Optimizing for `X` requires solving a system of linear equations with a `T x T`, so if your `T` (number of time ticks in data) is huge, you should use this flag to avoid memory errors. But if `T` is moderate (say, in the hundreds or first thousands), you're better of using dense matrices, because they work faster, so don't use this flag. Note, that if your lag set is'nt very big, there's probably no point in using very large T, as the temporal model is quite primitive and doesn't need too much data to train.
* `-b` - with this flag, the output matrix include the recovered values for Y, so the shape of output will be `n x (T + horizon)`; without this flag only the predictd values are output and so the shape will be  `n x horizon`;
* `-x lambda_x` - the value for the regularization coefficient `lambda_x`. The corresponding term in the optimization objective penalizes the rows of X disagreeing with the autoregression model. For example, `-x 1000`;
* -`w lambda_w` - the value for the regularization coefficient `lambda_w`. The corresponding term in the optimization objective penalizes large autoregression coefficients. For example, `-w 100`;
* -`f lambda_f` - the value for the regularization coefficient `lambda_f`. The corresponding term in the optimization objective penalizes large coefficients in matrix `F`, which defines the linear mapping from `X`-space  to `Y`-space. For example, `-f 0.01`;
* -`e eta` - the value for the regularization coefficient `eta`. The corresponding term in the optimization objective penalizes large coefficients in matrix `X`. For example, `-e 0.01`;

The resulting matrix is saved in CSV format again, each line corresponding to one time series.

## Getting data

this program comes with a dataset of cryptocurrencies exchanges rate, you can [download it here](https://drive.google.com/file/d/1qDxtI_sPWtIwhuJq92_1-3cv7ecJmsD0/view?usp=sharing). Put the `data` firectory in the same location where the executive file is.

The files are in a rather peculiar format, you can `converter.py` script, included in this repo, to convert these files to CSV and combine a set of columns from different files into a single file. The script is written in Python 3 and requies Pandas.

Usage:
`converter.py -i file_with_list -o output_file_name -b begin_datetime -e end_datetime -c columns_list`-s standardize

* `-i file_with_list` - path to a file containing paths to the files that you want to include un the table. An example is of such file is included, it will work if `data` directory is in the same directory that yor executive file is.
* `-b begin_datetime` the time moment you want to start your dataset to start with. If cone cof the files doesnt cover this moment it will be excluded from the result and a corresponding message will be printed. THe datetime should be in the format
 `yyyy.mm.dd.hh.mm.ss`. 
* `-e end_datetime`. Same format as above, if the moment isnt covered, the file will be excluded.
* `-c columns_list`. Column names that you want to include, separated by spaces. For example, `O C` .
* `-s` - add this flag if you want to standardize every time series (useful mostly for vizualization purposes)

Example:
`-i file_list -o converted.csv -b 2016.12.29.00.00.00 -e 2018.02.01.00.00.00 -c O C`

This will produce three files:
*`converted.csv` - the data matrix per se, where each row is a time series;
* `converted_series_csv` - the row index (which row corresponds to which series from which file)
* `converted_time.csv` - the columns index (which column coressponds to which moment in time) 

######################################################################################################3

# Report

## Implementation details

The following main subroutines were developed:

* Two versions of Ridge Regression for the `W`-step, one based on SVD an another on the Cholesky decomposition. The two functions appear to have similar perfomance, but Cholesky should be faster in theory, so it was used in the end.

* Two versions of the `X`-step (solvedusing the Conjugate Gradients algorithm), one dense and one sparse. For the moderately sized data we have, the dense routines proved to be faster by a factor of about 2, but the sparse routines are left as an option that can be necessary for bigger data.  To switch to sparse version, `-s` flag is used. 

* `F`-step aka coordinate descent algorithm.

The program works as follows:

1. The data is loaded into a Eigen Matrix. Any value in the file that can't be converted to a `double` with `std::stod()` is seen as a missing value, and the missing mask `Omega` is filled simultaneously with `Y`. 
2. Every row of matrix Y is standardized, with scales and shifts stored for future inverse transformation.
3. The data matrix is elementwise multiplied by the mask Omega, so the missing values are set to zero. This is necessary for correct work of the X-stepm although the autors fail to mention that in their paper).
2. First approximations for `F` and `X` are obtained with a simple SVD decomposition.
3. The iterations are run, starting with an `W`-step, until the mean squared difference in the elements of `X` are less than  a hard value 0.0001, but no more than 20 iterations (an 'iteration' meaning the whole three steps).
5. The values in X for `horizon` time steps are predicted using the autoregression model. 
6. If the flag `-b` was used, `F` is multiplied by the whole matrix `X` to obtain a `n x (T+horizon)` reconstruction and prediction matrix, otherwise `F` is multiplied by the predicted part of `X` (the `horizon` right columns).
7. The matrix factorization approach to forecasting doesnt guarantee that the predicted values start where the known values end. This is solvedby simply shifting all the reconstucted series, so that value at last time tick from data match.
8. The data is scaled and shifted back to original scale using the previously stored scales and shifts.

## Experiments

The natural question to consider, is of course, the choice of regularization coefficients. Recall that there there are four of them: `lambda_w`, `lambda_f`, `lambda_x` and `eta`. One should remember that `eta` is the coefficient for the Frobenius norm of X, and this term is inside the general `X` term, which has its own coefficient `lambda_X`, so `eta` is effectively scaled by the factor of `lambda_X` The problem of choosing the right values proves to be very nontrivial, and the behaviour of factorization varies wildly depending on what the coefficients are. 

A more or less complete study of this question requires more time than we actually have, but we could deduce some vagueideasof what the coefficients should be.

* The higher `lambda_w`, the smoother the reconstructed/predicted ccurve is. Setting `lambda_w` >> 1000 practically amounts to having a constant prediction.

* The ratio `lambda_x` /  `lambda_w` should be >> 1

* `lambda_f` should be << 1

Using the following parameters on the dataset with daily data for 101 currencvy pairs

`rank 32; lags 1,2,3,4,5,6,7,14,21; horizon 25; lambda_x 10000;  lambda_w = 1000 lambda_f 0.01, eta 0.001`

gives the following picture: ![](https://i.imgur.com/SVkE05c.png)

The algorithm is trained on the data left to blue line, and then predicts 25 ticks to the right.

It appears that more often than not, the algorithm can predict the general direction of the exchange rate. But it seems that it just kinda picks up the general trend and extrapolates it.

But let's try to use the same values, but on a different time period of the same data:

![](https://i.imgur.com/pHLL1BL.png)

Now the prediction don't seem reasonable.

With some random search< Ifound the following combination of parameters, that seems to work well here:
rank 32; lags 1,2,3,4,5,6,7,14,21; horizon 25; lambda_x 100; lambda_w 10; f 0.01; eta 0.001

![](https://i.imgur.com/5vPWpEB.png)

Trying the same set of values on a different set of series we again obtain do

![](https://i.imgur.com/T4ZeaG9.png)

We tried many more parameter combinations, but the results were similarly inconclusive.

Conclusion:

* The TRMF is very finicky about the choice of hyperparameters. 

* The TRMF is not adaptive enough for such volatile data as exchange rates, let alone cryptocurrence exchange rates! I think that one can use the algorithm for this purpose, but not in its vanilla autoregressive variant, and not without som adaptive behaviour.

* For some more regular data maybe TRMF can be enough.

## Comparison to other method.

We tried to use a method, suggested by Ilya Trofimov, where the time prediction problem is transformed into a supervised learning problem. The idea is that for a fixed lag set, object-target pairs are formed, where target is series[i], and the object [series[i-lag1], ... ,series[i-lagL], series_id]. This dataset is then fed into catboost with series-id as a categorical feature. I have no idea why, but catboost just would not train on such data, stating

`Training has stopped (degenerate solution on iteration 0, probably too small l2-regularization, try to increase it)`.

I tried chainging the l2 regularization coefficient, taking less tuples into the dataset, changing th learning rate and downgrading catboost - nothing helped. It would train with no more than 125 data points, and 126 points already cause the error above.

We hope we will overcome this difficulty ot use other algotihms and add a meaninful comparison to this report later, before the hard deadline.
