# Common
import copy
import json
import os
import re
import random
import math

# Data manipulation
import numpy as np

# Machine learning
from sklearn.linear_model import LinearRegression, LogisticRegression, Perceptron
from sklearn.tree import DecisionTreeRegressor, ExtraTreeRegressor, DecisionTreeClassifier, ExtraTreeClassifier
from sklearn.svm import SVR, SVC
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, RandomForestClassifier
from sklearn.neural_network import MLPRegressor
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.cross_decomposition import PLSRegression
from skmultiflow.trees import RegressionHoeffdingTree, RegressionHAT, HoeffdingTree, HAT
from lib.bagging import BaggingRegression
from lib.genetic_selector import GeneticSelector
from lib.genetic_selection_cv import GeneticSelectionCV
from lib.sklearn_relief import ReliefF, RReliefF
from sklearn.feature_selection import SelectKBest, f_regression, f_classif
from sklearn.preprocessing import MinMaxScaler, KBinsDiscretizer
from sklearn.metrics import mean_squared_error, r2_score, f1_score
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV, TimeSeriesSplit, KFold

# Local
from .log import Log as log



# Regression machine learning methods
ML_METHODS_REG = [
    # Batch
    {
        'name': 'LinearRegression',
        'description': 'Linear Regression',
        'params': {},
    },
    {
        'name': 'DecisionTreeRegressor',
        'description': 'Decision Tree Regressor',
        'params': {
            'random_state': "eval:random_seed"
        },
        'params_grid': {
            'splitter': ['random', 'best'],
            'max_depth': [2, 3, 5, 7, None],
            # 'max_features': ['sqrt', 'auto', 'auto', None]
        },
    },
    {
        'name': 'RandomForestRegressor',
        'description': 'Random Forest Regressor',
        'params': {
            'n_estimators': 10,
            'random_state': "eval:random_seed"
        },
        'params_grid': {
            'max_depth': [2, 3, 5, 7, None],
            # 'max_features': ['sqrt', 'auto', 'auto', None],
            'n_estimators': [int(x) for x in np.linspace(start = 10, stop = 200, num = 10)]
        },
    },
    {
        'name': 'GradientBoostingRegressor',
        'description': 'Gradient Boosting Regressor',
        'params': {
            'random_state': "eval:random_seed"
        },
        'params_grid': {
            'max_depth': [2, 3, 5, 7, None],
            # 'max_features': ['sqrt', 'auto', 'auto', None],
            'n_estimators': [int(x) for x in np.linspace(start = 10, stop = 200, num = 10)]
        },
    },
    {
        'name': 'PLSRegression',
        'description': 'Partial Least Squares Regression',
        'params': {},
    },
    {
        'name': 'ExtraTreeRegressor',
        'description': 'Extremely Randomized Tree Regressor',
        'params': {
            'random_state': "eval:random_seed"
        },
        'params_grid': {
            'splitter': ['random', 'best'],
            'max_depth': [2, 3, 5, 7, None],
            # 'max_features': ['sqrt', 'auto', 'auto', None]
        },
    },
    {
        'name': 'SVR',
        'description': 'Epsilon-Support Vector Regression',
        'params': {
            'gamma': 'auto',
        },
        'scaler': {
            'name': 'MinMaxScaler',
        },
    },
    {
        'name': 'MLPRegressor',
        'description': 'Multi-Layer Perceptron Regressor',
        'params': {
            'random_state': "eval:random_seed"
        },
        'params_grid': {
            'hidden_layer_sizes': [int(x) for x in np.linspace(start = 10, stop = 200, num = 10)]
        },
        'scaler': {
            'name': 'MinMaxScaler',
        },
    },
    {
        'name': 'KNeighborsRegressor',
        'description': 'k-Nearest Neighbors Regressor',
        'params': {},
        'params_grid': {
            'n_neighbors': [3, 5, 7, 13]
        },
        'scaler': {
            'name': 'MinMaxScaler',
        },
    },
    # Streaming
    {
        'name': 'RegressionHoeffdingTree',
        'description': 'Hoeffding Tree Regressor',
        'params': {
            'leaf_prediction': 'perceptron',
            'random_state': "eval:random_seed"
        },
        'params_grid': {
            # 'leaf_prediction': ['perceptron', 'mean'],
            'grace_period': [10, 25, 50, 100, 200, 500],
            'binary_split': [True, False],
            'no_preprune': [True, False],
            'nb_threshold': [0, 1, 2, 3, 4, 5],
        },
    },
    {
        'name': 'RegressionHAT',
        'description': 'Hoeffding Adaptive Tree Regressor',
        'params': {
            'leaf_prediction': 'perceptron',
            'random_state': "eval:random_seed"
        },
        'params_grid': {
            # 'leaf_prediction': ['perceptron', 'mean'],
            'grace_period': [10, 25, 50, 100, 200, 500],
            'binary_split': [True, False],
            'no_preprune': [True, False],
            'nb_threshold': [0, 1, 2, 3, 4, 5],
        }
    },
    {
        'name': 'BaggingRegression',
        'description': 'Hoeffding Tree Bagging Regressor',
        'params': {
            'base_estimator': {
                'name': "RegressionHoeffdingTree",
                'params': {
                    'leaf_prediction': 'perceptron',
                    'random_state': "eval:random_seed"
                },
            },
        },
    },
    {
        'name': 'BaggingRegression',
        'description': 'Hoeffding Adaptive Tree Bagging Regressor',
        'params': {
            'base_estimator': {
                'name': "RegressionHAT",
                'params': {
                    'leaf_prediction': 'perceptron',
                    'random_state': "eval:random_seed"
                },
            },
        },
    },
]

# Classification machine learning methods
ML_METHODS_CLS = [
    # Batch
    {
        'name': 'LogisticRegression',
        'description': 'Logistic Regression',
        'params': {
            'solver': 'liblinear',
            'multi_class': 'auto',
            'random_state': "eval:random_seed"
        },
    },
    {
        'name': 'DecisionTreeClassifier',
        'description': 'Decision Tree Classifier',
        'params': {
            'random_state': "eval:random_seed"
        },
        'params_grid': {
            'splitter': ['random', 'best'],
            'max_depth': [2, 3, 5, 7, None],
            # 'max_features': ['sqrt', 'auto', 'auto', None]
        },
    },
    {
        'name': 'ExtraTreeClassifier',
        'description': 'Extremely Randomized Tree Classifier',
        'params': {
            'random_state': "eval:random_seed"
        },
        'params_grid': {
            'splitter': ['random', 'best'],
            'max_depth': [2, 3, 5, 7, None],
            # 'max_features': ['sqrt', 'auto', 'auto', None]
        },
    },
    {
        'name': 'RandomForestClassifier',
        'description': 'Random Forest Classifier',
        'params': {
            'random_state': "eval:random_seed",
            'n_estimators': 10
        },
        'params_grid': {
            'max_depth': [2, 3, 5, 7, None],
            # 'max_features': ['sqrt', 'auto', 'auto', None],
            'n_estimators': [int(x) for x in np.linspace(start = 10, stop = 200, num = 10)]
        },
    },
    {
        'name': 'SVC',
        'description': 'C-Support Vector Classification',
        'params': {
            'gamma': 'auto',
            'random_state': "eval:random_seed"
        },
        'scaler': {
            'name': 'MinMaxScaler',
        },
    },
    {
        'name': 'KNeighborsClassifier',
        'description': 'k-Nearest Neighbors Classifier',
        'params': {},
        'params_grid': {
            'n_neighbors': [3, 5, 7, 13]
        },
        'scaler': {
            'name': 'MinMaxScaler',
        },
    },
    {
        'name': 'Perceptron',
        'description': 'Perceptron Classifier',
        'params': {
            'random_state': "eval:random_seed"
        },
        'scaler': {
            'name': 'MinMaxScaler',
        },
    },
    # Streaming
    {
        'name': 'GaussianNB',
        'description': 'Gaussian Naïve Bayes Classifier',
        'params': {},
    },
    {
        'name': 'HoeffdingTree',
        'description': 'Hoeffding Tree Classifier',
        'params': {},
        'params_grid': {
            # 'leaf_prediction': ['mc', 'nb', 'nba'],
            'grace_period': [10, 25, 50, 100, 200, 500],
            'binary_split': [True, False],
            'no_preprune': [True, False],
            'nb_threshold': [0, 1, 2, 3, 4, 5],
        }
    },
    {
        'name': 'HAT',
        'description': 'Hoeffding Adaptive Tree Classifier',
        'params': {},
        'params_grid': {
            # 'leaf_prediction': ['mc', 'nb', 'nba'],
            'grace_period': [10, 25, 50, 100, 200, 500],
            'binary_split': [True, False],
            'no_preprune': [True, False],
            'nb_threshold': [0, 1, 2, 3, 4, 5],
        }
    },
]

# Regression evaluation methods
EVAL_METHODS_REG = [
    {
        'name': 'r2_score',
        'description': 'Coefficient of Determination',
        'params': {},
    },
    {
        'name': 'mean_squared_error',
        'description': 'Mean Squared Error',
        'params': {
            # 'squared': False
        },
    },
    {
        'name': 'root_mean_squared_error',
        'description': 'Root Mean Squared Error',
        'params': {},
    },
]

# Classification evaluation methods
EVAL_METHODS_CLS = [
    {
        'name': 'f1_score',
        'description': 'F1 Score',
        'params': {
            'average': 'macro'
        },
    },
]

# Feature selection methods
FS_METHODS = [
    {
        'name': 'SelectKBest',
        'description': 'Feature Selection according to the highest scores',
        'params': {
            'score_func': "eval:f_regression",
            'k': 20
        }
    },
    {
        'name': 'GeneticSelector',
        'description': 'Feature Selection with Genetic Algorithm',
        'params': {
            # 'estimator': {
            #     "name": "LinearRegression",
            # },
            'estimator': None,
            # 'n_gen': 7,
            'n_gen': 5,
            # 'size': 200,
            'size': 100,
            # 'n_best': 40,
            'n_best': 20,
            # 'n_rand': 40, 
            'n_rand': 20, 
            # 'n_children': 5,
            'n_children': 5,
            'mutation_rate': 0.05,
            'max_features': 30,
        }
    },
    {
        'name': 'GeneticSelectionCV',
        'description': 'Feature Selection with Genetic Algorithm',
        'params': {
            'estimator': None,
            'cv': 5,
            'verbose': 1,
            'scoring': "accuracy",
            'max_features': 30,
            'n_population': 50,
            'crossover_proba': 0.5,
            'mutation_proba': 0.2,
            'n_generations': 40,
            'crossover_independent_proba': 0.5,
            'mutation_independent_proba': 0.05,
            'tournament_size': 3,
            'n_gen_no_change': 10,
            'caching': True,
            'n_jobs': -1
        },
    },
    {
        'name': 'ReliefF',
        'description': 'Feature selection for classification with ReliefF',
        'params': {
            'k': 10,
            'approx_decimals': 4,
            'ramp': False,
            'n_iterations': 100,
            'n_features': 20,
            'random_state': 'eval:random_seed'
        }
    },
    {
        'name': 'RReliefF',
        'description': 'Feature selection for regression with RReliefF',
        'params': {
            'sigma': 0.1,
            'k': 10,
            'approx_decimals': 4,
            'ramp': False,
            'n_iterations': 100,
            'n_features': 30,
            'random_state': 'eval:random_seed'
        }
    }
]

# Hyper parameter optimization methods
HPO_METHODS = [
    {
        'name': 'RandomizedSearchCV',
        'description': 'Randomized Search with Cross-Validation',
        'params': {
            'estimator': None,
            'param_distributions': None,
            'n_iter': 20,
            'cv': 3,
            'verbose': 2,
            'random_state': 42,
            'n_jobs': 1
        }
    },
    {
        'name': 'GridSearchCV',
        'description': 'Grid Search with Cross-Validation',
        'params': {
            # 'estimator': {
            #     "name": "LinearRegression",
            # },
            'estimator': None,
            'param_grid': None,
            'cv': 3,
            'verbose': 2,
            'n_jobs': 1
        }
    },
]


# Cross-Validation methods
CV_METHODS = [
    {
        'name': 'TimeSeriesSplit',
        'description': 'Time Series cross-validator',
        'params': {
            'n_splits': None, # Dynamically set.
            'max_train_size': None
        }
    },
    {
        'name': 'KFold',
        'description': 'K-Folds cross-validator',
        'params': {
            'n_splits': None, # Dynamically set.
            'shuffle': False
        }
    }
]

# Discretization methods
DIS_METHODS = [
    {
        'name': 'KBinsDiscretizer',
        'description': 'Discretization with Binning into k Intervals',
        'params': {
            'n_bins': 8,
            'encode': 'ordinal',
            # 'strategy': 'uniform',
            'strategy': 'quantile',
            # 'strategy': 'kmeans',
        }
    }
]


def call(name, params):
    return eval(name)(**params)


def eval_obj(obj, **kwargs):
    if isinstance(obj, dict):
        for k, v in obj.items():
            obj[k] = eval_obj(v, **kwargs)
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            obj[i] = eval_obj(v, **kwargs)
    else:
        if isinstance(obj, str):
            match = re.match(r'^(eval\:)(.+)', obj)
            if match:
                for k, v in kwargs.items():
                    exec(f'{k} = {v}')
                obj = eval(match.group(2))

    return obj


def find_item(name, items):
    for item in items:
        if item['name'] == name:
            return item
    
    return None


def map_items(items, ref_items):
    ref_dict = { i['name']: i for i in ref_items }
    return [ref_dict[i] if isinstance(i, str) else i for i in items]


def root_mean_squared_error(*args, **kwargs):
    return math.sqrt(mean_squared_error(*args, **kwargs))


def r2_adj_score(y_true, y_pred, n_features, sample_weight=None, multioutput="uniform_average"):
    r2 = r2_score(y_true, y_pred, sample_weight, multioutput)
    n_samples = len(y_true)

    return r2 - (n_features - 1) / (n_samples - n_features) * (1 - r2)



class ComplexEncoder(json.JSONEncoder):
    def default(self, obj):
        try:
            return json.JSONEncoder.default(self, obj)
        except:
            return obj.__name__


class Config():
    def __init__(
        self,
        water_type, # 'ground' or 'surface'
        sensors,
        date_from,
        date_to,
        features,
        max_average,
        max_shift,
        ml_methods,
        eval_methods,
        fs_method,
        hpo_method,
        cv_method,
        dis_method,
        horizon,
        random_seed
    ):
        # Map machine learning methods.
        if ml_methods:
            ml_methods_reg = ml_methods.get('reg')
            if ml_methods_reg:
                ml_methods['reg'] = map_items(ml_methods_reg, ML_METHODS_REG)
            
            ml_methods_cls = ml_methods.get('cls')
            if ml_methods_cls:
                ml_methods['cls'] = map_items(ml_methods_cls, ML_METHODS_CLS)

        # Map evaluation methods.
        if eval_methods:
            eval_methods_reg = eval_methods.get('reg')
            if eval_methods_reg:
                eval_methods['reg'] = map_items(eval_methods_reg, EVAL_METHODS_REG)
            
            eval_methods_cls = eval_methods.get('cls')
            if eval_methods_cls:
                eval_methods['cls'] = map_items(eval_methods_cls, EVAL_METHODS_CLS)

        # Map feature selection method.
        if fs_method and isinstance(fs_method, str):
            fs_method = find_item(fs_method, FS_METHODS)

        # Map hyperparameter optimization method.
        if hpo_method and isinstance(hpo_method, str):
            hpo_method = find_item(hpo_method, HPO_METHODS)

        # Map cross-validation method.
        if cv_method and isinstance(cv_method, str):
            cv_method = find_item(cv_method, CV_METHODS)

        # Map discretization method.
        if dis_method and isinstance(dis_method, str):
            dis_method = find_item(dis_method, DIS_METHODS)

        # Random seed
        if random_seed:
            random.seed(random_seed)
            np.random.seed(random_seed)

        self.config = eval_obj({
            'water_type': water_type,
            'sensors': sensors,
            'date_from': date_from,
            'date_to': date_to,
            'features': features,
            'max_average': max_average,
            'max_shift': max_shift,
            'ml_methods': ml_methods,
            'eval_methods': eval_methods,
            'fs_method': fs_method,
            'hpo_method': hpo_method,
            'cv_method': cv_method,
            'dis_method': dis_method,
            'horizon': horizon,
            'random_seed': random_seed,
        }, random_seed=random_seed)


    def save(self, path):
        file_path = os.path.join(path, 'config.json')
        log.info(f'Saving configuration: {file_path}')
        with open(file_path, 'w', encoding="utf-8") as file:
            json.dump(self.config, file, cls=ComplexEncoder)


    def get(self, key=''):
        if not key:
            return self.config
        elif key not in self.config:
            log.warning(f'Option \'{key}\' does not exist.')
            return None
        else:
            return self.config[key]


def main():
    pass


if __name__ == '__main__':
    main()
