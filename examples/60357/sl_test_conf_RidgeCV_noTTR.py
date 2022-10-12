# Testing the removal of TransformedTargetRegressor
# from the SuperLearner. It's a good idea to transform
# the target for skewed distributions:
# https://scikit-learn.org/stable/auto_examples/compose/plot_transformed_target.html#sphx-glr-auto-examples-compose-plot-transformed-target-py
# Also use RidgeCV for the stacking regressor.

from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.base import BaseEstimator,RegressorMixin

from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.linear_model import RidgeCV
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

# Remove for testing
#from sklearn.compose import TransformedTargetRegressor
#from scipy.optimize import nnls

from scipy.stats import loguniform,uniform,randint
import numpy as np
import pickle

# NEEDS:
# skopt 0.8.1 (scikit-opt)
# sklearn 0.23.2
# xboost 1.3.3

# DEFINE SUPERLEARNER
#---------------Temporarily removed---------------------
#class NonNegativeLeastSquares(BaseEstimator, RegressorMixin):
#    def __init__(self):
#        pass
#
#    def fit(self, X, y):
#        X, y = check_X_y(X, y)
#        self.weights_, _ = nnls(X, y)
#        return self
#
#    def predict(self, X):
#        check_is_fitted(self)
#        X = check_array(X)
#        return np.matmul(X, self.weights_)
#----------End Temporarily removed----------------------

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
# data as inputs to ML.
#
# Note that the target's transformer does NOT need to be
# the same as the input transformer.  In all cases, retain
# the target transformer as MinMaxScaler.

SuperLearnerConf = {
    "final_estimator": Pipeline(
        [
            ('scale', StandardScaler()),
            ('final',RidgeCV())
        ]
    ),
    "estimators": {
        "nusvr-rbf": {
            "model": Pipeline(
                [
                    ('scale', StandardScaler()),
                    ('svr', NuSVR(kernel='rbf'))
                ]
            ),
            "hpo": BayesSearchCV(
                Pipeline(
                    [
                        ('scale', StandardScaler()),
                        ('svr', NuSVR(kernel='rbf'))
                    ]
                ),
                {
                    "svr__C": (10**-6, 10**2.5, 'log-uniform'),
                    "svr__nu": (10**-10, 0.99, 'uniform'),
                    "svr__gamma": (10**-6, 0.99, 'log-uniform')
                },
                n_iter = n_iter,
                cv = cv
            )
        },
        "nusvr-lin": {
            "model": Pipeline(
                [
                    ('scale', MinMaxScaler()),
                    ('svr', NuSVR(kernel='linear'))
                ]
            ),
            "hpo": BayesSearchCV(
                Pipeline(
                    [
                        ('scale', MinMaxScaler()),
                        ('svr', NuSVR(kernel='linear'))
                    ]
                ),
                {
                    "svr__C": (10**-6, 10**2.5, 'log-uniform'),
                    "svr__nu": (10**-10, 0.99, 'uniform')
                },
                n_iter = n_iter,
                cv = cv
            )
        },
        "nusvr-poly": {
            "model": Pipeline(
                [
                    ('scale', MinMaxScaler()),
                    ('svr', NuSVR(kernel='poly'))
                ]
            ),
            "hpo": BayesSearchCV(
                Pipeline(
                    [
                        ('scale', MinMaxScaler()),
                        ('svr', NuSVR(kernel='poly'))
                    ]
                ),
                {
                    "svr__C": (10**-6, 10**2.5, 'log-uniform'),
                    "svr__nu": (10**-10, 0.99, 'uniform'),
                    "svr__degree": [1, 2, 3]
                },
                n_iter = n_iter,
                cv = cv
            )
        },
        "nusvr-sig": {
            "model": Pipeline(
                [
                    ('scale', MinMaxScaler()),
                    ('svr', NuSVR(kernel='sigmoid'))
                ]
            ),
            "hpo": BayesSearchCV(
                Pipeline(
                    [
                        ('scale', MinMaxScaler()),
                        ('svr', NuSVR(kernel='sigmoid'))
                    ]
                ),
                {
                    "svr__C": (10**-6, 10**2.5, 'log-uniform'),
                    "svr__nu": (10**-10, 0.99, 'uniform'),
                    "svr__coef0": [-0.99, 0.99, 'uniform']
                },
                n_iter = n_iter,
                cv = cv
            )
        },
        "knn-uni": {
            "model": Pipeline(
                [
                    ('scale', MinMaxScaler()),
                    ('knn', KNeighborsRegressor(weights='uniform'))
                ]
            ),
            "hpo": BayesSearchCV(
                Pipeline(
                    [
                        ('scale', MinMaxScaler()),
                        ('knn', KNeighborsRegressor(weights='uniform'))
                    ]
                ),
                {
                    "knn__n_neighbors": (1, 10, 'uniform')
                },
                n_iter = n_iter,
                cv = cv
            )
        },
        "knn-dist": {
            "model": Pipeline(
                [
                    ('scale', MinMaxScaler()),
                    ('knn', KNeighborsRegressor(weights='distance'))
                ]
            ),
            "hpo": BayesSearchCV(
                Pipeline(
                    [
                        ('scale', MinMaxScaler()),
                        ('knn', KNeighborsRegressor(weights='distance'))
                    ]
                ),
                {
                    "knn__n_neighbors": (1, 10, 'uniform')
                },
                n_iter = n_iter,
                cv = cv
            )
        },
        "pls": {
            "model": Pipeline(
                [
                    ('scale', MinMaxScaler()),
                    ('plsr', PLSRegression())
                ]
            ),
            "hpo": BayesSearchCV(
                Pipeline(
                    [
                        ('scale', MinMaxScaler()),
                        ('plsr', PLSRegression())
                    ]
                ),
                {
                    "plsr__n_components": (1, 10, 'uniform')
                },
                n_iter = n_iter,
                cv = cv
            )
        },
        "mlp": {
            "model": Pipeline(
                [
                    ('scale',  MinMaxScaler()),
                    ('mlp', MLPRegressor())
                ]
            ),
            "hpo": BayesSearchCV(
                Pipeline(
                    [
                        ('scale',  MinMaxScaler()),
                        ('mlp', MLPRegressor())
                    ]
                ),
                {
                    "mlp__hidden_layer_sizes": (10, 250),
                    "mlp__solver": ["lbfgs", "sgd", "adam"],
                    "mlp__alpha": (10**-6, 0.99, 'log-uniform'),
                    "mlp__tol": (10**-6, 10**-2, 'log-uniform')
                },
                n_iter = n_iter,
                cv = cv
            )
        },
        "ridge": {
            "model": Pipeline(
                [
                    ('scale',  StandardScaler()),
                    ('poly', PolynomialFeatures(degree = 3)),
                    ('linear', Ridge())
                ]
            ),
            "hpo": BayesSearchCV(
                Pipeline(
                    [
                        ('scale',  StandardScaler()),
                        ('poly', PolynomialFeatures(degree = 3)),
                        ('linear', Ridge())
                    ]
                ),
                {
                    "poly__degree": [1, 2, 3],
                    "linear__alpha": (10**-6, 0.99, 'log-uniform')
                },
                n_iter = n_iter,
                cv = cv
            )
        },
        "lasso": {
            "model": Pipeline(
                [
                    ('scale',  StandardScaler()),
                    ('poly', PolynomialFeatures(degree = 3)),
                    ('linear', Ridge())
                ]
            ),
            "hpo": BayesSearchCV(
                Pipeline(
                    [
                        ('scale',  StandardScaler()),
                        ('poly', PolynomialFeatures(degree = 3)),
                        ('linear', Lasso())
                    ]
                ),
                {
                    "poly__degree": [1, 2, 3],
                    "linear__alpha": (10**-6, 0.99, 'log-uniform')
                },
                n_iter = n_iter,
                cv = cv
            )
        },
        "linear": {
            "model": Pipeline(
                [
                    ('scale',  MinMaxScaler()),
                    ('poly', PolynomialFeatures(degree = 3)),
                    ('linear', LinearRegression())
                ]
            ),
            "hpo": BayesSearchCV(
                Pipeline(
                    [
                        ('scale',  MinMaxScaler()),
                        ('poly', PolynomialFeatures(degree = 3)),
                        ('linear', LinearRegression())
                    ]
                ),
                {
                    "poly__degree": [1, 2, 3]
                },
                n_iter = n_iter,
                cv = cv
            )
        },
        "enet": {
            "model": Pipeline(
                [
                    ('scale',  StandardScaler()),
                    ('poly', PolynomialFeatures(degree = 3)),
                    ('linear', ElasticNet())
                ]
            ),
            "hpo": BayesSearchCV(
                Pipeline(
                    [
                        ('scale',  StandardScaler()),
                        ('poly', PolynomialFeatures(degree = 3)),
                        ('linear', ElasticNet())
                    ]
                ),
                {
                    "poly__degree": [1, 2, 3],
                    "linear__alpha": (10**-6, 0.99, 'log-uniform'),
                    "linear__l1_ratio": (10**-6, 0.99, 'log-uniform')
                },
                n_iter = n_iter,
                cv = cv
            )
        },
        "huber": {
            "model": Pipeline(
                [
                    ('scale',  StandardScaler()),
                    ('poly', PolynomialFeatures(degree = 3)),
                    ('linear', HuberRegressor())
                ]
            ),
            "hpo": BayesSearchCV(
                Pipeline(
                    [
                        ('scale',  StandardScaler()),
                        ('poly', PolynomialFeatures(degree = 3)),
                        ('linear', HuberRegressor())
                    ]
                ),
                {
                    "poly__degree": [1, 2, 3],
                    "linear__alpha": (10**-6, 0.99, 'log-uniform'),
                    "linear__epsilon": (1.35, 1.9, 'uniform')
                },
                n_iter = n_iter,
                cv = cv
            )
        },
        "xgb": {
            "model": Pipeline(
                [
                    ('scale',  MinMaxScaler()),
                    ('xgb', XGBRegressor(objective = 'reg:squarederror'))
                ]
            ),
            "hpo": BayesSearchCV(
                Pipeline(
                    [
                        ('scale',  MinMaxScaler()),
                        ('xgb', XGBRegressor(objective = 'reg:squarederror'))
                    ]
                ),
                {
                    "xgb__n_estimators": (100, 10000),
                    "xgb__learning_rate": (10**-4, 0.99, 'log-uniform'),
                    "xgb__max_depth": [2, 3, 4, 5, 6, 7, 8]
                },
                n_iter = n_iter,
                cv = cv
            )
        },
        "etr": {
            "model": Pipeline(
                [
                    ('scale',  MinMaxScaler()),
                    ('etr', ExtraTreesRegressor())
                ]
            ),
            "hpo": BayesSearchCV(
                Pipeline(
                    [
                        ('scale',  MinMaxScaler()),
                        ('etr', ExtraTreesRegressor())
                    ]
                ),
                {
                    "etr__n_estimators": (100, 10000),
                    "etr__ccp_alpha": [0, 0.001, 0.01, 0.1],
                    "etr__max_features": ["auto", "sqrt", "log2"],
                    "etr__criterion": ["mse", "mae"],
                    "etr__max_depth": [2, 3, 4, 5, 6, 7, 8],
                    "etr__min_samples_split": [0.1, 0.2, 0.3],
                    "etr__min_samples_leaf": [0.1, 0.2, 0.3]
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
