from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.base import BaseEstimator,RegressorMixin

from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
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
                    "regressor__etr__max_features": ["auto", "sqrt", "log2"],
                    "regressor__etr__criterion": ["mse", "mae"],
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


# FOR BACKUP:
"""
# Sample with RndomizedSearchCV
SuperLearnerConf_RandomizedSearchCV = {
    "final_estimator": NonNegativeLeastSquares(),
    "estimators": {
        "nusvr": {
            "model": Pipeline(
                [
                    ('scale', MinMaxScaler()),
                    ('svr', NuSVR())
                ]
            ),
            "hpo": RandomizedSearchCV(
                Pipeline(
                    [
                        ('scale', MinMaxScaler()),
                        ('svr', NuSVR())
                    ]
                ),
                {
                    "svr__C": loguniform(10**-6, 10**2.5),
                    "svr__nu": uniform(1e-10, 0),
                    "svr__gamma": loguniform(10**-6, 1)
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
            "hpo": RandomizedSearchCV(
                Pipeline(
                    [
                        ('scale',  MinMaxScaler()),
                        ('mlp', MLPRegressor())
                    ]
                ),
                {
                    "mlp__hidden_layer_sizes": [(10,),  (20,),  (40,),
                                           (60,),  (80,),  (100,),
                                           (120,), (140,), (160,),
                                           (180,), (200,), (220,),
                                           (240,)],
                    "mlp__alpha": loguniform(10**-6, 10**0),
                    "mlp__tol": loguniform(10**-6, 10**-2)
                },
                n_iter = n_iter,
                cv = cv
            )
        },
        "ridge": {
            "model": Pipeline(
                [
                    ('scale',  MinMaxScaler()),
                    ('poly', PolynomialFeatures(degree = 3)),
                    ('linear', Ridge())
                ]
            ),
            "hpo": RandomizedSearchCV(
                Pipeline(
                    [
                        ('scale',  MinMaxScaler()),
                        ('poly', PolynomialFeatures(degree = 3)),
                        ('linear', Ridge())
                    ]
                ),
                {
                    "poly__degree": [1, 2, 3],
                    "linear__alpha": loguniform(10**-6, 10**0)
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
            "hpo":  RandomizedSearchCV(
                Pipeline(
                    [
                        ('scale',  MinMaxScaler()),
                        ('xgb', XGBRegressor(objective = 'reg:squarederror'))
                    ]
                ),
                {
                    "xgb__n_estimators": [100, 200, 400, 800, 1000,
                                          2000, 3000, 4000, 5000, 6000,
                                          7000, 8000, 9000, 10000],
                    "xgb__learning_rate": loguniform(10**-4, 10**0),
                    "xgb__max_depth": [2, 3, 4, 5, 6, 7, 8]
                },
                n_iter = n_iter,
                cv = cv
            )
        }
    }
}
"""
