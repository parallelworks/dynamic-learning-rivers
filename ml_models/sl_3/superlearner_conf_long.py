# SuperLearner configuration for:
# 15 stacked ensemble models
# Each uses MinMaxScaler or StandardScaler on inputs
# Each uses TransformedTargetRegressor with custom log10
# and MinMaxScaler function/inverse pair on targets.
#
# This configuration is an attempt to reproduce as
# closely as possible the simple approach to taking
# a log10 of the target and training then model on
# the transformed data.

from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.base import BaseEstimator,RegressorMixin

from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import PowerTransformer
from sklearn.linear_model import Ridge
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.linear_model import ElasticNet
from sklearn.linear_model import HuberRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.cross_decomposition import PLSRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
from xgboost import XGBRegressor
from sklearn.svm import NuSVR
from sklearn.kernel_ridge import KernelRidge
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import RandomizedSearchCV
from skopt import BayesSearchCV
from skopt.space import Real, Categorical, Integer

from sklearn.compose import TransformedTargetRegressor

from scipy.optimize import nnls
from scipy.stats import loguniform,uniform,randint
import numpy as np
import pickle

# NEEDS:
# skopt 0.8.1 (scikit-opt)
# sklearn 0.23.2
# xboost 1.3.3

# DEFINE SUPERLEARNER
class NonNegativeLeastSquares(BaseEstimator, RegressorMixin):
    def __init__(self):
        pass

    def fit(self, X, y):
        X, y = check_X_y(X, y)
        self.weights_, _ = nnls(X, y)
        return self

    def predict(self, X):
        check_is_fitted(self)
        X = check_array(X)
        return np.matmul(X, self.weights_)


# _neg values apply -1 assuming all input values
# are negative, so the log operation can work on
# positive values.

# Manually specify MinMaxScaler bookends.  Since
# the log10 operation is applied before the MMS,
# these represent the span of orders of magnitude.
mms_min = -2.0
mms_max = 4.0

def mms_log10_neg(input):

    #print(mms_min)
    #print(mms_max)
    
    # MinMaxScaler should have all values between 0 and 1.
    max_val = 1
    min_val = 0
    
    # Apply log10 transform
    do_log10 = np.log10(-1.0*input)

    # Apply MinMaxScaler
    output = (do_log10 - mms_min)/(mms_max - mms_min)

    # Count bad values
    #num_too_big=np.sum(np.sum(output > max_val))
    #num_too_small=np.sum(np.sum(output < min_val))
    #print('Transform: Num too big: '+str(num_too_big)+' Num too small: '+str(num_too_small))

    # Filter values
    output[output > max_val] = max_val
    output[output < min_val] = min_val

    return output

    #print('Function: '+str(np.sum(np.sum(output > max_val))))
    #print('Inp Min: '+str(np.min(input))+' Max: '+str(np.max(input)))
    #print('Out Min: '+str(np.min(output))+' Max: '+str(np.max(output)))
    #print('Nans: '+str(np.sum(np.sum(np.isnan(output)))))
    #max_val = np.log1p(np.finfo(np.float64).max/2)
    #output = np.log1p(np.abs(input))
    #print(np.sum(np.sum(output > max_val)))
    #output[output < -1.0*max_val]=-1.0*max_val
    #output[output < 0.0] = 0.0
    #output[output > max_val] = max_val
    #return output

def neg_exp10_mms(input):
    #print(mms_min)
    #print(mms_max)

    #max_val = np.finfo(np.float32).max
    #min_val = np.finfo(np.float32).eps

    # Prefilter before apply MinMaxScaler
    input[input < 0.0] = 0
    input[input > 1.0] = 1

    # Undo MinMaxScaler
    undo_mms = input*(mms_max-mms_min) + mms_min

    # Undo the log10 transform
    output = (10.0**undo_mms)

    # Check output is reasonable
    #num_too_big=np.sum(np.sum(output > max_val))
    #num_too_small=np.sum(np.sum(output < min_val))
    #print('Inverse: Num too big: '+str(num_too_big)+' Num too small: '+str(num_too_small))
    #output[output > max_val] = max_val
    #output[output < min_val] = min_val
    #output[np.isnan(output)] = min_val
    #output[np.isinf(output)] = max_val

    # Make all values negative.
    output = -1.0*output
    
    return output

    #print('Inverse: '+str(np.sum(np.sum(output > max_val))))
    #print('Inp Min: '+str(np.min(input))+' Max: '+str(np.max(input)))
    #print('Out Min: '+str(np.min(output))+' Max: '+str(np.max(output)))
    #print('Nans: '+str(np.sum(np.sum(np.isnan(output)))))
    #max_val = np.log1p(np.finfo(np.float64).max/2)
    #input_copy = input
    #input_copy[input_copy > max_val] = max_val
    #input_copy[input_copy < -1.0*max_val] = -1.0*max_val
    #output = -1.0*np.expm1(input_copy)
    #return output

n_iter = 10
cv = 5

# About using the HPO as a model:
# 1. Is is very expensive. The SL uses cross-validation and so does the HPO.
# 2. Does it even make sense to run cross-validation inside the cross-validation?
# 3. Parallelism fails locally (works with dask). Probably using joblib inside joblib?
# 4. If HPO is activated the best model is passed as the model to the SL

# MinMaxScaler is default scaler for pipelines except for
# nusvr-rbf and linear models with regularization terms
# (Ridge, Lasso, Elastic-Net, Huber):
# https://scikit-learn.org/stable/modules/preprocessing.html
# In those cases, use StandardScaler to get centered, Gaussian
# data as inputs to ML. Pipelines only apply to INPUT (i.e. ML
# model features).
#
# The TransformedTargetRegressor allows for integrating a
# transform on the target in each model. Note that the 
# target's transformer does NOT need to be
# the same as the input transformer.

SuperLearnerConf = {
    "final_estimator": NonNegativeLeastSquares(),
    "estimators": {
        "nusvr-rbf": {
            "model": TransformedTargetRegressor(
                regressor = Pipeline(
                    [
                        ('scale', StandardScaler()),
                        ('svr', NuSVR(kernel='rbf'))
                    ]
                ),
                transformer = MinMaxScaler()
            ),
            "hpo": BayesSearchCV(
                TransformedTargetRegressor(
                    regressor = Pipeline(
                        [
                            ('scale', StandardScaler()),
                            ('svr', NuSVR(kernel='rbf'))
                        ]
                    ),
                    transformer = MinMaxScaler()
                ),
                {
                    "regressor__svr__C": (10**-6, 10**2.5, 'log-uniform'),
                    "regressor__svr__nu": (10**-10, 0.99, 'uniform'),
                    "regressor__svr__gamma": (10**-6, 0.99, 'log-uniform')
                },
                n_iter = n_iter,
                cv = cv
            )
        },
        "nusvr-lin": {
            "model": TransformedTargetRegressor(
                regressor = Pipeline(
                    [
                        ('scale', MinMaxScaler()),
                        ('svr', NuSVR(kernel='linear'))
                    ]
                ),
                transformer = MinMaxScaler()
            ),
            "hpo": BayesSearchCV(
                TransformedTargetRegressor(
                    regressor = Pipeline(
                        [
                            ('scale', MinMaxScaler()),
                            ('svr', NuSVR(kernel='linear'))
                        ]
                    ),
                    transformer = MinMaxScaler()
                ),
                {
                    "regressor__svr__C": (10**-6, 10**2.5, 'log-uniform'),
                    "regressor__svr__nu": (10**-10, 0.99, 'uniform')
                },
                n_iter = n_iter,
                cv = cv
            )
        },
        "nusvr-poly": {
            "model": TransformedTargetRegressor(
                regressor = Pipeline(
                    [
                        ('scale', MinMaxScaler()),
                        ('svr', NuSVR(kernel='poly'))
                    ]
                ),
                transformer = MinMaxScaler()
            ),
            "hpo": BayesSearchCV(
                TransformedTargetRegressor(
                    regressor = Pipeline(
                        [
                            ('scale', MinMaxScaler()),
                            ('svr', NuSVR(kernel='poly'))
                        ]
                    ),
                    transformer = MinMaxScaler()
                ),
                {
                    "regressor__svr__C": (10**-6, 10**2.5, 'log-uniform'),
                    "regressor__svr__nu": (10**-10, 0.99, 'uniform'),
                    "regressor__svr__degree": [1, 2, 3]
                },
                n_iter = n_iter,
                cv = cv
            )
        },
        "nusvr-sig": {
            "model": TransformedTargetRegressor(
                regressor = Pipeline(
                    [
                        ('scale', MinMaxScaler()),
                        ('svr', NuSVR(kernel='sigmoid'))
                    ]
                ),
                transformer = MinMaxScaler()
            ),
            "hpo": BayesSearchCV(
                TransformedTargetRegressor(
                    regressor = Pipeline(
                        [
                            ('scale', MinMaxScaler()),
                            ('svr', NuSVR(kernel='sigmoid'))
                        ]
                    ),
                    transformer = MinMaxScaler()
                ),
                {
                    "regressor__svr__C": (10**-6, 10**2.5, 'log-uniform'),
                    "regressor__svr__nu": (10**-10, 0.99, 'uniform'),
                    "regressor__svr__coef0": [-0.99, 0.99, 'uniform']
                },
                n_iter = n_iter,
                cv = cv
            )
        },
        "knn-uni": {
            "model": TransformedTargetRegressor(
                regressor = Pipeline(
                    [
                        ('scale', MinMaxScaler()),
                        ('knn', KNeighborsRegressor(weights='uniform'))
                    ]
                ),
                transformer = MinMaxScaler()
            ),
            "hpo": BayesSearchCV(
                TransformedTargetRegressor(
                    regressor = Pipeline(
                        [
                            ('scale', MinMaxScaler()),
                            ('knn', KNeighborsRegressor(weights='uniform'))
                        ]
                    ),
                    transformer = MinMaxScaler()
                ),
                {
                    "regressor__knn__n_neighbors": (1, 10, 'uniform')
                },
                n_iter = n_iter,
                cv = cv
            )
        },
        "knn-dist": {
            "model": TransformedTargetRegressor(
                regressor = Pipeline(
                    [
                        ('scale', MinMaxScaler()),
                        ('knn', KNeighborsRegressor(weights='distance'))
                    ]
                ),
                transformer = MinMaxScaler()
            ),
            "hpo": BayesSearchCV(
                TransformedTargetRegressor(
                    regressor = Pipeline(
                        [
                            ('scale', MinMaxScaler()),
                            ('knn', KNeighborsRegressor(weights='distance'))
                        ]
                    ),
                    transformer = MinMaxScaler()
                ),
                {
                    "regressor__knn__n_neighbors": (1, 10, 'uniform')
                },
                n_iter = n_iter,
                cv = cv
            )
        },
        "pls": {
            "model": TransformedTargetRegressor(
                regressor = Pipeline(
                    [
                        ('scale', MinMaxScaler()),
                        ('plsr', PLSRegression())
                    ]
                ),
                transformer = MinMaxScaler()
            ),
            "hpo": BayesSearchCV(
                TransformedTargetRegressor(
                    regressor = Pipeline(
                        [
                            ('scale', MinMaxScaler()),
                            ('plsr', PLSRegression())
                        ]
                    ),
                    transformer = MinMaxScaler()
                ),
                {
                    "regressor__plsr__n_components": (1, 10, 'uniform')
                },
                n_iter = n_iter,
                cv = cv
            )
        },
        "mlp": {
            "model": TransformedTargetRegressor(
                regressor = Pipeline(
                    [
                        ('scale',  MinMaxScaler()),
                        ('mlp', MLPRegressor())
                    ]
                ),
                transformer = MinMaxScaler()
            ),
            "hpo": BayesSearchCV(
                TransformedTargetRegressor(
                    regressor = Pipeline(
                        [
                            ('scale',  MinMaxScaler()),
                            ('mlp', MLPRegressor())
                        ]
                    ),
                    transformer = MinMaxScaler()
                ),
                {
                    "regressor__mlp__hidden_layer_sizes": (10, 250),
                    "regressor__mlp__solver": ["lbfgs", "sgd", "adam"],
                    "regressor__mlp__alpha": (10**-6, 0.99, 'log-uniform'),
                    "regressor__mlp__tol": (10**-6, 10**-2, 'log-uniform')
                },
                n_iter = n_iter,
                cv = cv
            )
        },
        "ridge": {
            "model": TransformedTargetRegressor(
                regressor = Pipeline(
                    [
                        ('scale',  StandardScaler()),
                        ('poly', PolynomialFeatures(degree = 3)),
                        ('linear', Ridge())
                    ]
                ),
                transformer = MinMaxScaler()
            ),
            "hpo": BayesSearchCV(
                TransformedTargetRegressor(
                    regressor = Pipeline(
                        [
                            ('scale',  StandardScaler()),
                            ('poly', PolynomialFeatures(degree = 3)),
                            ('linear', Ridge())
                        ]
                    ),
                    transformer = MinMaxScaler()
                ),
                {
                    "regressor__poly__degree": [1, 2, 3],
                    "regressor__linear__alpha": (10**-6, 0.99, 'log-uniform')
                },
                n_iter = n_iter,
                cv = cv
            )
        },
        "lasso": {
            "model": TransformedTargetRegressor(
                regressor = Pipeline(
                    [
                        ('scale',  StandardScaler()),
                        ('poly', PolynomialFeatures(degree = 3)),
                        ('linear', Lasso())
                    ]
                ),
                transformer = MinMaxScaler()
            ),
            "hpo": BayesSearchCV(
                TransformedTargetRegressor(
                    regressor = Pipeline(
                        [
                            ('scale',  StandardScaler()),
                            ('poly', PolynomialFeatures(degree = 3)),
                            ('linear', Lasso())
                        ]
                    ),
                    transformer = MinMaxScaler()
                ),
                {
                    "regressor__poly__degree": [1, 2, 3],
                    "regressor__linear__alpha": (10**-6, 0.99, 'log-uniform')
                },
                n_iter = n_iter,
                cv = cv
            )
        },
        "linear": {
            "model": TransformedTargetRegressor(
                regressor = Pipeline(
                    [
                        ('scale',  MinMaxScaler()),
                        ('poly', PolynomialFeatures(degree = 3)),
                        ('linear', LinearRegression())
                    ]
                ),
                transformer = MinMaxScaler()
            ),
            "hpo": BayesSearchCV(
                TransformedTargetRegressor(
                    regressor = Pipeline(
                        [
                            ('scale',  MinMaxScaler()),
                            ('poly', PolynomialFeatures(degree = 3)),
                            ('linear', LinearRegression())
                        ]
                    ),
                    transformer = MinMaxScaler()
                ),
                {
                    "regressor__poly__degree": [1, 2, 3]
                },
                n_iter = n_iter,
                cv = cv
            )
        },
        "enet": {
            "model": TransformedTargetRegressor(
                regressor = Pipeline(
                    [
                        ('scale',  StandardScaler()),
                        ('poly', PolynomialFeatures(degree = 3)),
                        ('linear', ElasticNet())
                    ]
                ),
                transformer = MinMaxScaler()
            ),
            "hpo": BayesSearchCV(
                TransformedTargetRegressor(
                    regressor = Pipeline(
                        [
                            ('scale',  StandardScaler()),
                            ('poly', PolynomialFeatures(degree = 3)),
                            ('linear', ElasticNet())
                        ]
                    ),
                    transformer = MinMaxScaler()
                ),
                {
                    "regressor__poly__degree": [1, 2, 3],
                    "regressor__linear__alpha": (10**-6, 0.99, 'log-uniform'),
                    "regressor__linear__l1_ratio": (10**-6, 0.99, 'log-uniform')
                },
                n_iter = n_iter,
                cv = cv
            )
        },
        "huber": {
            "model": TransformedTargetRegressor(
                regressor = Pipeline(
                    [
                        ('scale',  StandardScaler()),
                        ('poly', PolynomialFeatures(degree = 3)),
                        ('linear', HuberRegressor())
                    ]
                ),
                transformer = MinMaxScaler()
            ),
            "hpo": BayesSearchCV(
                TransformedTargetRegressor(
                    regressor = Pipeline(
                        [
                            ('scale',  StandardScaler()),
                            ('poly', PolynomialFeatures(degree = 3)),
                            ('linear', HuberRegressor())
                        ]
                    ),
                    transformer = MinMaxScaler()
                ),
                {
                    "regressor__poly__degree": [1, 2, 3],
                    "regressor__linear__alpha": (10**-6, 0.99, 'log-uniform'),
                    "regressor__linear__epsilon": (1.35, 1.9, 'uniform')
                },
                n_iter = n_iter,
                cv = cv
            )
        },
        "xgb": {
            "model": TransformedTargetRegressor(
                regressor = Pipeline(
                    [
                        ('scale',  MinMaxScaler()),
                        ('xgb', XGBRegressor(objective = 'reg:squarederror'))
                    ]
                ),
                transformer = MinMaxScaler()
            ),
            "hpo": BayesSearchCV(
                TransformedTargetRegressor(
                    regressor = Pipeline(
                        [
                            ('scale',  MinMaxScaler()),
                            ('xgb', XGBRegressor(objective = 'reg:squarederror'))
                        ]
                    ),
                    transformer = MinMaxScaler()
                ),
                {
                    "regressor__xgb__n_estimators": (100, 10000),
                    "regressor__xgb__learning_rate": (10**-4, 0.99, 'log-uniform'),
                    "regressor__xgb__max_depth": [2, 3, 4, 5, 6, 7, 8]
                },
                n_iter = n_iter,
                cv = cv
            )
        },
        "etr": {
            "model": TransformedTargetRegressor(
                regressor = Pipeline(
                    [
                        ('scale',  MinMaxScaler()),
                        ('etr', ExtraTreesRegressor())
                    ]
                ),
                transformer = MinMaxScaler()
            ),
            "hpo": BayesSearchCV(
                TransformedTargetRegressor(
                    regressor = Pipeline(
                        [
                            ('scale',  MinMaxScaler()),
                            ('etr', ExtraTreesRegressor())
                        ]
                    ),
                    transformer = MinMaxScaler()
                ),
                {
                    "regressor__etr__n_estimators": (100, 10000),
                    "regressor__etr__ccp_alpha": [0, 0.001, 0.01, 0.1],
                    "regressor__etr__max_features": [0.1, 0.3, 0.5, 0.8, 1.0],
                    "regressor__etr__criterion": ["squared_error", "absolute_error", "friedman_mse", "poisson"],
                    "regressor__etr__max_depth": [2, 3, 4, 5, 6, 7, 8],
                    "regressor__etr__min_samples_split": [0.1, 0.2, 0.3],
                    "regressor__etr__min_samples_leaf": [0.1, 0.2, 0.3]
                },
                n_iter = n_iter,
                cv = cv
            )
        },
        "nusvr-rbf-log10": {
            "model": TransformedTargetRegressor(
                regressor = Pipeline(
                    [
                        ('scale', StandardScaler()),
                        ('svr', NuSVR(kernel='rbf'))
                    ]
                ),
                func=mms_log10_neg, inverse_func=neg_exp10_mms, check_inverse=True
            ),
            "hpo": BayesSearchCV(
                TransformedTargetRegressor(
                    regressor = Pipeline(
                        [
                            ('scale', StandardScaler()),
                            ('svr', NuSVR(kernel='rbf'))
                        ]
                    ),
                    func=mms_log10_neg, inverse_func=neg_exp10_mms, check_inverse=True
                ),
                {
                    "regressor__svr__C": (10**-6, 10**2.5, 'log-uniform'),
                    "regressor__svr__nu": (10**-10, 0.99, 'uniform'),
                    "regressor__svr__gamma": (10**-6, 0.99, 'log-uniform')
                },
                n_iter = n_iter,
                cv = cv
            )
        },
        "nusvr-lin-log10": {
            "model": TransformedTargetRegressor(
                regressor = Pipeline(
                    [
                        ('scale', MinMaxScaler()),
                        ('svr', NuSVR(kernel='linear'))
                    ]
                ),
                func=mms_log10_neg, inverse_func=neg_exp10_mms, check_inverse=True
            ),
            "hpo": BayesSearchCV(
                TransformedTargetRegressor(
                    regressor = Pipeline(
                        [
                            ('scale', MinMaxScaler()),
                            ('svr', NuSVR(kernel='linear'))
                        ]
                    ),
                    func=mms_log10_neg, inverse_func=neg_exp10_mms, check_inverse=True
                ),
                {
                    "regressor__svr__C": (10**-6, 10**2.5, 'log-uniform'),
                    "regressor__svr__nu": (10**-10, 0.99, 'uniform')
                },
                n_iter = n_iter,
                cv = cv
            )
        },
        "nusvr-poly-log10": {
            "model": TransformedTargetRegressor(
                regressor = Pipeline(
                    [
                        ('scale', MinMaxScaler()),
                        ('svr', NuSVR(kernel='poly'))
                    ]
                ),
                func=mms_log10_neg, inverse_func=neg_exp10_mms, check_inverse=True
            ),
            "hpo": BayesSearchCV(
                TransformedTargetRegressor(
                    regressor = Pipeline(
                        [
                            ('scale', MinMaxScaler()),
                            ('svr', NuSVR(kernel='poly'))
                        ]
                    ),
                    func=mms_log10_neg, inverse_func=neg_exp10_mms, check_inverse=True
                ),
                {
                    "regressor__svr__C": (10**-6, 10**2.5, 'log-uniform'),
                    "regressor__svr__nu": (10**-10, 0.99, 'uniform'),
                    "regressor__svr__degree": [1, 2, 3]
                },
                n_iter = n_iter,
                cv = cv
            )
        },
        "nusvr-sig-log10": {
            "model": TransformedTargetRegressor(
                regressor = Pipeline(
                    [
                        ('scale', MinMaxScaler()),
                        ('svr', NuSVR(kernel='sigmoid'))
                    ]
                ),
                func=mms_log10_neg, inverse_func=neg_exp10_mms, check_inverse=True
            ),
            "hpo": BayesSearchCV(
                TransformedTargetRegressor(
                    regressor = Pipeline(
                        [
                            ('scale', MinMaxScaler()),
                            ('svr', NuSVR(kernel='sigmoid'))
                        ]
                    ),
                    func=mms_log10_neg, inverse_func=neg_exp10_mms, check_inverse=True
                ),
                {
                    "regressor__svr__C": (10**-6, 10**2.5, 'log-uniform'),
                    "regressor__svr__nu": (10**-10, 0.99, 'uniform'),
                    "regressor__svr__coef0": [-0.99, 0.99, 'uniform']
                },
                n_iter = n_iter,
                cv = cv
            )
        },
        "knn-uni-log10": {
            "model": TransformedTargetRegressor(
                regressor = Pipeline(
                    [
                        ('scale', MinMaxScaler()),
                        ('knn', KNeighborsRegressor(weights='uniform'))
                    ]
                ),
                func=mms_log10_neg, inverse_func=neg_exp10_mms, check_inverse=True
            ),
            "hpo": BayesSearchCV(
                TransformedTargetRegressor(
                    regressor = Pipeline(
                        [
                            ('scale', MinMaxScaler()),
                            ('knn', KNeighborsRegressor(weights='uniform'))
                        ]
                    ),
                    func=mms_log10_neg, inverse_func=neg_exp10_mms, check_inverse=True
                ),
                {
                    "regressor__knn__n_neighbors": (1, 10, 'uniform')
                },
                n_iter = n_iter,
                cv = cv
            )
        },
        "knn-dist-log10": {
            "model": TransformedTargetRegressor(
                regressor = Pipeline(
                    [
                        ('scale', MinMaxScaler()),
                        ('knn', KNeighborsRegressor(weights='distance'))
                    ]
                ),
                func=mms_log10_neg, inverse_func=neg_exp10_mms, check_inverse=True
            ),
            "hpo": BayesSearchCV(
                TransformedTargetRegressor(
                    regressor = Pipeline(
                        [
                            ('scale', MinMaxScaler()),
                            ('knn', KNeighborsRegressor(weights='distance'))
                        ]
                    ),
                    func=mms_log10_neg, inverse_func=neg_exp10_mms, check_inverse=True
                ),
                {
                    "regressor__knn__n_neighbors": (1, 10, 'uniform')
                },
                n_iter = n_iter,
                cv = cv
            )
        },
        "pls-log10": {
            "model": TransformedTargetRegressor(
                regressor = Pipeline(
                    [
                        ('scale', MinMaxScaler()),
                        ('plsr', PLSRegression())
                    ]
                ),
                func=mms_log10_neg, inverse_func=neg_exp10_mms, check_inverse=True
            ),
            "hpo": BayesSearchCV(
                TransformedTargetRegressor(
                    regressor = Pipeline(
                        [
                            ('scale', MinMaxScaler()),
                            ('plsr', PLSRegression())
                        ]
                    ),
                    func=mms_log10_neg, inverse_func=neg_exp10_mms, check_inverse=True
                ),
                {
                    "regressor__plsr__n_components": (1, 10, 'uniform')
                },
                n_iter = n_iter,
                cv = cv
            )
        },
        "mlp-log10": {
            "model": TransformedTargetRegressor(
                regressor = Pipeline(
                    [
                        ('scale',  MinMaxScaler()),
                        ('mlp', MLPRegressor())
                    ]
                ),
                func=mms_log10_neg, inverse_func=neg_exp10_mms, check_inverse=True
            ),
            "hpo": BayesSearchCV(
                TransformedTargetRegressor(
                    regressor = Pipeline(
                        [
                            ('scale',  MinMaxScaler()),
                            ('mlp', MLPRegressor())
                        ]
                    ),
                    func=mms_log10_neg, inverse_func=neg_exp10_mms, check_inverse=True
                ),
                {
                    "regressor__mlp__hidden_layer_sizes": (10, 250),
                    "regressor__mlp__solver": ["lbfgs", "sgd", "adam"],
                    "regressor__mlp__alpha": (10**-6, 0.99, 'log-uniform'),
                    "regressor__mlp__tol": (10**-6, 10**-2, 'log-uniform')
                },
                n_iter = n_iter,
                cv = cv
            )
        },
        "ridge-log10": {
            "model": TransformedTargetRegressor(
                regressor = Pipeline(
                    [
                        ('scale',  StandardScaler()),
                        ('poly', PolynomialFeatures(degree = 3)),
                        ('linear', Ridge())
                    ]
                ),
                func=mms_log10_neg, inverse_func=neg_exp10_mms, check_inverse=True
            ),
            "hpo": BayesSearchCV(
                TransformedTargetRegressor(
                    regressor = Pipeline(
                        [
                            ('scale',  StandardScaler()),
                            ('poly', PolynomialFeatures(degree = 3)),
                            ('linear', Ridge())
                        ]
                    ),
                    func=mms_log10_neg, inverse_func=neg_exp10_mms, check_inverse=True
                ),
                {
                    "regressor__poly__degree": [1, 2, 3],
                    "regressor__linear__alpha": (10**-6, 0.99, 'log-uniform')
                },
                n_iter = n_iter,
                cv = cv
            )
        },
        "lasso-log10": {
            "model": TransformedTargetRegressor(
                regressor = Pipeline(
                    [
                        ('scale',  StandardScaler()),
                        ('poly', PolynomialFeatures(degree = 3)),
                        ('linear', Lasso())
                    ]
                ),
                func=mms_log10_neg, inverse_func=neg_exp10_mms, check_inverse=True
            ),
            "hpo": BayesSearchCV(
                TransformedTargetRegressor(
                    regressor = Pipeline(
                        [
                            ('scale',  StandardScaler()),
                            ('poly', PolynomialFeatures(degree = 3)),
                            ('linear', Lasso())
                        ]
                    ),
                    func=mms_log10_neg, inverse_func=neg_exp10_mms, check_inverse=True
                ),
                {
                    "regressor__poly__degree": [1, 2, 3],
                    "regressor__linear__alpha": (10**-6, 0.99, 'log-uniform')
                },
                n_iter = n_iter,
                cv = cv
            )
        },
        "linear-log10": {
            "model": TransformedTargetRegressor(
                regressor = Pipeline(
                    [
                        ('scale',  MinMaxScaler()),
                        ('poly', PolynomialFeatures(degree = 3)),
                        ('linear', LinearRegression())
                    ]
                ),
                func=mms_log10_neg, inverse_func=neg_exp10_mms, check_inverse=True
            ),
            "hpo": BayesSearchCV(
                TransformedTargetRegressor(
                    regressor = Pipeline(
                        [
                            ('scale',  MinMaxScaler()),
                            ('poly', PolynomialFeatures(degree = 3)),
                            ('linear', LinearRegression())
                        ]
                    ),
                    func=mms_log10_neg, inverse_func=neg_exp10_mms, check_inverse=True
                ),
                {
                    "regressor__poly__degree": [1, 2, 3]
                },
                n_iter = n_iter,
                cv = cv
            )
        },
        "enet-log10": {
            "model": TransformedTargetRegressor(
                regressor = Pipeline(
                    [
                        ('scale',  StandardScaler()),
                        ('poly', PolynomialFeatures(degree = 3)),
                        ('linear', ElasticNet())
                    ]
                ),
                func=mms_log10_neg, inverse_func=neg_exp10_mms, check_inverse=True
            ),
            "hpo": BayesSearchCV(
                TransformedTargetRegressor(
                    regressor = Pipeline(
                        [
                            ('scale',  StandardScaler()),
                            ('poly', PolynomialFeatures(degree = 3)),
                            ('linear', ElasticNet())
                        ]
                    ),
                    func=mms_log10_neg, inverse_func=neg_exp10_mms, check_inverse=True
                ),
                {
                    "regressor__poly__degree": [1, 2, 3],
                    "regressor__linear__alpha": (10**-6, 0.99, 'log-uniform'),
                    "regressor__linear__l1_ratio": (10**-6, 0.99, 'log-uniform')
                },
                n_iter = n_iter,
                cv = cv
            )
        },
        "huber-log10": {
            "model": TransformedTargetRegressor(
                regressor = Pipeline(
                    [
                        ('scale',  StandardScaler()),
                        ('poly', PolynomialFeatures(degree = 3)),
                        ('linear', HuberRegressor())
                    ]
                ),
                func=mms_log10_neg, inverse_func=neg_exp10_mms, check_inverse=True
            ),
            "hpo": BayesSearchCV(
                TransformedTargetRegressor(
                    regressor = Pipeline(
                        [
                            ('scale',  StandardScaler()),
                            ('poly', PolynomialFeatures(degree = 3)),
                            ('linear', HuberRegressor())
                        ]
                    ),
                    func=mms_log10_neg, inverse_func=neg_exp10_mms, check_inverse=True
                ),
                {
                    "regressor__poly__degree": [1, 2, 3],
                    "regressor__linear__alpha": (10**-6, 0.99, 'log-uniform'),
                    "regressor__linear__epsilon": (1.35, 1.9, 'uniform')
                },
                n_iter = n_iter,
                cv = cv
            )
        },
        "xgb-log10": {
            "model": TransformedTargetRegressor(
                regressor = Pipeline(
                    [
                        ('scale',  MinMaxScaler()),
                        ('xgb', XGBRegressor(objective = 'reg:squarederror'))
                    ]
                ),
                func=mms_log10_neg, inverse_func=neg_exp10_mms, check_inverse=True
            ),
            "hpo": BayesSearchCV(
                TransformedTargetRegressor(
                    regressor = Pipeline(
                        [
                            ('scale',  MinMaxScaler()),
                            ('xgb', XGBRegressor(objective = 'reg:squarederror'))
                        ]
                    ),
                    func=mms_log10_neg, inverse_func=neg_exp10_mms, check_inverse=True
                ),
                {
                    "regressor__xgb__n_estimators": (100, 10000),
                    "regressor__xgb__learning_rate": (10**-4, 0.99, 'log-uniform'),
                    "regressor__xgb__max_depth": [2, 3, 4, 5, 6, 7, 8]
                },
                n_iter = n_iter,
                cv = cv
            )
        },
        "etr-log10": {
            "model": TransformedTargetRegressor(
                regressor = Pipeline(
                    [
                        ('scale',  MinMaxScaler()),
                        ('etr', ExtraTreesRegressor())
                    ]
                ),
                func=mms_log10_neg, inverse_func=neg_exp10_mms, check_inverse=True
            ),
            "hpo": BayesSearchCV(
                TransformedTargetRegressor(
                    regressor = Pipeline(
                        [
                            ('scale',  MinMaxScaler()),
                            ('etr', ExtraTreesRegressor())
                        ]
                    ),
                    func=mms_log10_neg, inverse_func=neg_exp10_mms, check_inverse=True
                ),
                {
                    "regressor__etr__n_estimators": (100, 10000),
                    "regressor__etr__ccp_alpha": [0, 0.001, 0.01, 0.1],
                    "regressor__etr__max_features": [0.1, 0.3, 0.5, 0.8, 1.0],
                    "regressor__etr__criterion": ["squared_error", "absolute_error", "friedman_mse", "poisson"],
                    "regressor__etr__max_depth": [2, 3, 4, 5, 6, 7, 8],
                    "regressor__etr__min_samples_split": [0.1, 0.2, 0.3],
                    "regressor__etr__min_samples_leaf": [0.1, 0.2, 0.3]
                },
                n_iter = n_iter,
                cv = cv
            )
        }
    }
}

def load_SuperLearner(model_pkl):
    with open(model_pkl, 'rb') as inp:
        return pickle.load(inp)

