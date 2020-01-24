# Utilities
import os
import json
import math
import random
import copy
import re
import time
import itertools
from datetime import datetime
from abc import ABCMeta
from lib.log import Log as log
from lib.config import Config
from lib.experiment import Experiment
from lib.utils import (
    get_scripts_path,
    save_json,
    get_range
)

# Data manipulation
import pandas as pd
import numpy as np
import scipy.cluster.hierarchy as sch

# Plotting
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from pandas.plotting import register_matplotlib_converters

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
from sklearn.feature_selection import SelectKBest, f_regression, f_classif
from sklearn.preprocessing import MinMaxScaler, KBinsDiscretizer
from sklearn.metrics import mean_squared_error, r2_score, f1_score
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV, TimeSeriesSplit, KFold


RANDOM_SEED = 1234

# Regression machine learning methods
ML_METHODS_REG = [
    # Batch
    {
        'name': 'LinearRegression',
        'description': 'Linear Regression',
        'class': LinearRegression,
        'params': {},
    },
    {
        'name': 'DecisionTreeRegressor',
        'description': 'Decision Tree Regressor',
        'class': DecisionTreeRegressor,
        'params': {
            'random_state': RANDOM_SEED
        },
        'params_grid': {
            'splitter': ['random', 'best'],
            'max_depth': [2, 3, 5, 7, None],
            'max_features': ['sqrt', 'auto', 'auto', None]
        },
    },
    {
        'name': 'RandomForestRegressor',
        'description': 'Random Forest Regressor',
        'class': RandomForestRegressor,
        'params': {
            'n_estimators': 10,
            'random_state': RANDOM_SEED
        },
        'params_grid': {
            'max_depth': [2, 3, 5, 7, None],
            'max_features': ['sqrt', 'auto', 'auto', None],
            'n_estimators': [int(x) for x in np.linspace(start = 10, stop = 200, num = 10)]
        },
    },
    {
        'name': 'GradientBoostingRegressor',
        'description': 'Gradient Boosting Regressor',
        'class': GradientBoostingRegressor,
        'params': {
            'random_state': RANDOM_SEED
        },
        'params_grid': {
            'max_depth': [2, 3, 5, 7, None],
            'max_features': ['sqrt', 'auto', 'auto', None],
            'n_estimators': [int(x) for x in np.linspace(start = 10, stop = 200, num = 10)]
        },
    },
    {
        'name': 'PLSRegression',
        'description': 'Partial Least Squares Regression',
        'class': PLSRegression,
        'params': {},
    },
    {
        'name': 'ExtraTreeRegressor',
        'description': 'Extremely Randomized Tree Regressor',
        'class': ExtraTreeRegressor,
        'params': {
            'random_state': RANDOM_SEED
        },
        'params_grid': {
            'splitter': ['random', 'best'],
            'max_depth': [2, 3, 5, 7, None],
            'max_features': ['sqrt', 'auto', 'auto', None]
        },
    },
    {
        'name': 'SVR',
        'description': 'Epsilon-Support Vector Regression',
        'class': SVR,
        'params': {
            'gamma': 'auto',
        },
        'scaler': {
            'class': MinMaxScaler,
        },
    },
    {
        'name': 'MLPRegressor',
        'description': 'Multi-Layer Perceptron Regressor',
        'class': MLPRegressor,
        'params': {
            'random_state': RANDOM_SEED
        },
        'params_grid': {
            'hidden_layer_sizes': [int(x) for x in np.linspace(start = 10, stop = 200, num = 10)]
        },
        'scaler': {
            'class': MinMaxScaler,
        },
    },
    {
        'name': 'KNeighborsRegressor',
        'description': 'k-Nearest Neighbors Regressor',
        'class': KNeighborsRegressor,
        'params': {},
        'params_grid': {
            'n_neighbors': [3, 5, 7, 13]
        },
        'scaler': {
            'class': MinMaxScaler,
        },
    },
    # Streaming
    {
        'name': 'RegressionHoeffdingTree',
        'description': 'Hoeffding Tree Regressor',
        'class': RegressionHoeffdingTree,
        'params': {
            'leaf_prediction': 'perceptron',
            'random_state': 0
        },
        'params_grid': {
            'leaf_prediction': ['nba', 'mean', 'perceptron'],
            'grace_period': [10, 25, 50, 100, 200, 500],
            'binary_split': [True, False],
            'no_preprune': [True, False],
            'nb_threshold': [0, 1, 2, 3, 4, 5],
        },
    },
    {
        'name': 'RegressionHAT',
        'description': 'Hoeffding Adaptive Tree Regressor',
        'class': RegressionHAT,
        'params': {
            'leaf_prediction': 'perceptron',
            'random_state': 0
        },
        'params_grid': {
            'leaf_prediction': ['nba', 'mean', 'perceptron'],
            'grace_period': [10, 25, 50, 100, 200, 500],
            'binary_split': [True, False],
            'no_preprune': [True, False],
            'nb_threshold': [0, 1, 2, 3, 4, 5],
        }
    },
    {
        'name': 'BaggingRegressionHT',
        'description': 'Hoeffding Tree Bagging Regressor',
        'class': BaggingRegression,
        'params': {
            'base_estimator': {
                'name': 'Hoeffding Tree',
                'class': RegressionHoeffdingTree,
                'params': {
                    'leaf_prediction': 'perceptron',
                    'random_state': 0
                },
            },
        },
    },
    {
        'name': 'BaggingRegressionHAT',
        'description': 'Hoeffding Adaptive Tree Bagging Regressor',
        'class': BaggingRegression,
        'params': {
            'base_estimator': {
                'class': RegressionHAT,
                'params': {
                    'leaf_prediction': 'perceptron',
                    'random_state': 0
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
        'class': LogisticRegression,
        'params': {
            'solver': 'liblinear',
            'multi_class': 'auto',
            'random_state': RANDOM_SEED
        },
    },
    {
        'name': 'DecisionTreeClassifier',
        'description': 'Decision Tree Classifier',
        'class': DecisionTreeClassifier,
        'params': {
            'random_state': RANDOM_SEED
        },
        'params_grid': {
            'splitter': ['random', 'best'],
            'max_depth': [2, 3, 5, 7, None],
            'max_features': ['sqrt', 'auto', 'auto', None]
        },
    },
    {
        'name': 'ExtraTreeClassifier',
        'description': 'Extremely Randomized Tree Classifier',
        'class': ExtraTreeClassifier,
        'params': {
            'random_state': RANDOM_SEED
        },
        'params_grid': {
            'splitter': ['random', 'best'],
            'max_depth': [2, 3, 5, 7, None],
            'max_features': ['sqrt', 'auto', 'auto', None]
        },
    },
    {
        'name': 'RandomForestClassifier',
        'description': 'Random Forest Classifier',
        'class': RandomForestClassifier,
        'params': {
            'random_state': RANDOM_SEED,
            'n_estimators': 10
        },
        'params_grid': {
            'max_depth': [2, 3, 5, 7, None],
            'max_features': ['sqrt', 'auto', 'auto', None],
            'n_estimators': [int(x) for x in np.linspace(start = 10, stop = 200, num = 10)]
        },
    },
    {
        'name': 'SVC',
        'description': 'C-Support Vector Classification',
        'class': SVC,
        'params': {
            'gamma': 'auto',
            'random_state': RANDOM_SEED
        },
        'scaler': {
            'class': MinMaxScaler,
        },
    },
    {
        'name': 'KNeighborsClassifier',
        'description': 'k-Nearest Neighbors Classifier',
        'class': KNeighborsClassifier,
        'params': {},
        'params_grid': {
            'n_neighbors': [3, 5, 7, 13]
        },
        'scaler': {
            'class': MinMaxScaler,
        },
    },
    {
        'name': 'Perceptron',
        'description': 'Perceptron Classifier',
        'class': Perceptron,
        'params': {
            'random_state': RANDOM_SEED
        },
        'scaler': {
            'class': MinMaxScaler,
        },
    },
    # Streaming
    {
        'name': 'GaussianNB',
        'description': 'Gaussian Naïve Bayes Classifier',
        'class': GaussianNB,
        'params': {},
    },
    {
        'name': 'HoeffdingTree',
        'description': 'Hoeffding Tree Classifier',
        'class': HoeffdingTree,
        'params': {},
    },
    {
        'name': 'HAT',
        'description': 'Hoeffding Adaptive Tree Classifier',
        'class': HAT,
        'params': {},
    },
]

# Regression evaluation methods
EVAL_METHODS_REG = [
    {
        'name': 'R2',
        'description': 'Coefficient of Determination',
        'def': r2_score,
        'params': {},
    },
    {
        'name': 'MSE',
        'description': 'Mean Squared Error',
        'def': mean_squared_error,
        'params': {
            # 'squared': False
        },
    },
]

# Classification evaluation methods
EVAL_METHODS_CLS = [
    {
        'name': 'F1',
        'description': 'F1 Score',
        'def': f1_score,
        'params': {
            'average': 'macro'
        },
    },
]

# Feature selection methods
FS_METHODS = {
    'kbest': {
        'name': 'SelectKBest',
        'description': 'Feature Selection according to the highest scores',
        'class': SelectKBest,
        'params': {
            'score_func': f_regression,
            'k': 20
        }
    },
    'genetic': {
        'name': 'GeneticSelector',
        'description': 'Feature Selection with Genetic Algorithm',
        'class': GeneticSelector,
        'params': {
            'estimator': LinearRegression(), 
            'n_gen': 7,
            'size': 200,
            'n_best': 40,
            'n_rand': 40, 
            'n_children': 5,
            'mutation_rate': 0.05
        }
    },
}

# Hyper parameter optimization methods
HPO_METHODS = {
    'random': {
        'name': 'RandomizedSearchCV',
        'description': 'Randomized Search with Cross-Validation',
        'class': RandomizedSearchCV,
        'params': {
            'estimator': None,
            'param_distributions': None,
            'n_iter': 10,
            'cv': 3,
            'verbose': 2,
            'random_state': 42,
            'n_jobs': 1
        }
    },
    'params_grid': {
        'name': 'GridSearchCV',
        'description': 'Grid Search with Cross-Validation',
        'class': GridSearchCV,
        'params': {
            'estimator': None,
            'param_grid': None,
            'cv': 3,
            'verbose': 2,
            'n_jobs': 1
        }
    },
}


# Cross-Validation methods
CV_METHODS = {
    'time_series': {
        'name': 'TimeSeriesSplit',
        'description': 'Time Series cross-validator',
        'class': TimeSeriesSplit,
        'params': {
            'n_splits': None, # Dynamically set to number of years
            'max_train_size': None
        }
    },
    'kfold': {
        'name': 'KFold',
        'description': 'K-Folds cross-validator',
        'class': KFold,
        'params': {
            'n_splits': None, # Dynamically set to number of years
            'shuffle': False
        }
    }
}

# Discretization methods
DISC_METHODS = {
    'kbins': {
        'name': 'KBinsDiscretizer',
        'description': '',
        'class': KBinsDiscretizer,
        'params': {
            'n_bins': 10,
            'encode': 'ordinal',
            'strategy': 'uniform'
        }
    }
}



def filter_items(keys, items):
    return list([item for item in items if item['name'] in keys])


def main():
    register_matplotlib_converters()

    experiments = [
        Experiment(
            'Surface water levels',
            'Modeling relative changes of surface water levels',
            Config(
                water_type='surface',
                sensors=[
                    # 1060, # Gornja Radgona I (Mura)
                    # 1070, # Petanjci (Mura)
                    # 1140, # Pristava I (Ščavnica)
                    # 1260, # Čentiba (Ledava)
                    # 1335, # Središče (Ivanjševski potok)
                    # 1355, # Hodoš I (Velika Krka)
                    # 2250, # Otiški Vrh I (Meža)
                    # 2432, # Muta I (Bistrica)
                    # 2530, # Ruta (Radoljna)
                    # 2620, # Loče (Dravinja)
                    # 2719, # Podlehnik I (Rogatnica)
                    # 2830, # Ranca (Pesnica)
                    # 2900, # Zamušani I (Pesnica)
                    # 3080, # Blejski most (Sava Dolinka)
                    # 3200, # Sveti Janez (Sava Bohinjka)
                    # 3250, # Bodešče (Sava Bohinjka)
                    # 3320, # Bohinjska Bistrica (Bistrica)
                    # 3400, # Mlino I (Jezernica)
                    # 3420, # Radovljica I (Sava)
                    # 3530, # Medno (Sava)
                    # 3570, # Šentjakob (Sava)
                    # 3725, # Hrastnik (Sava)
                    # 3850, # Čatež I (Sava)
                    # 3900, # Jesenice na Dolenjskem (Sava)
                    # 4200, # Suha I (Sora)
                    # 4230, # Zminec (Poljanska Sora)
                    # 4270, # Železniki (Selška Sora)
                    # 4450, # Domžale (Mlinščica-Kanal)
                    # 4515, # Vir (Rača)
                    # 4520, # Podrečje (Rača)
                    # 4570, # Topole (Pšata)
                    # 4575, # Loka (Pšata)
                    # 4626, # Zagorje II (Medija)
                    # 4650, # Žebnik (Sopota)
                    # 4770, # Sodna vas II (Mestinjščica)
                    # 4860, # Metlika (Kolpa)
                    # 5040, # Kamin (Ljubljanica)
                    5078, # Moste I (Ljubljanica)
                    # 5330, # Borovnica (Borovniščica)
                    # 5425, # Iška vas (Iška)
                    # 5500, # Dvor (Gradaščica)
                    # 5880, # Hasberg (Unica)
                    # 5910, # Malni (Malenščica)
                    # 6060, # Nazarje (Savinja)
                    # 6068, # Letuš I (Savinja)
                    # 6200, # Laško I (Savinja)
                    # 6220, # Luče (Lučnica)
                    # 6300, # Šoštanj (Paka)
                    # 6340, # Rečica (Paka)
                    # 6350, # Škale (Lepena)
                    # 6385, # Pesje IV (Lepena)
                    # 6400, # Škale (Sopota)
                    # 6420, # Šoštanj (Velunja)
                    # 6550, # Dolenja vas II (Bolska)
                    # 6691, # Črnolica I (Voglajna)
                    # 6835, # Vodiško I (Gračnica)
                    # 7488, # Prigorica I (Ribnica)
                    # 8080, # Kobarid I (Soča)
                    # 8180, # Solkan I (Soča)
                    # 8230, # Log pod Mangartom (Koritnica)
                    # 8242, # Kal-Koritnica I (Koritnica)
                    # 8270, # Žaga (Učja)
                    # 8454, # Cerkno III (Cerknica)
                    # 8565, # Dolenje (Vipava)
                    # 8710, # Potoki (Nadiža)
                    # 9015, # Trpčane (Reka)
                    # 9030, # Trnovo (Reka)
                ],
                date_from='2010-01-01',
                date_to='2017-12-31',
                features=[
                    'day_time',
                    'precipitation',
                    'snow_accumulation',
                    'temperature_avg',
                    'temperature_min',
                    'temperature_max',
                    'cloud_cover_avg',
                    'cloud_cover_min',
                    'cloud_cover_max',
                    'dew_point_avg', 
                    'dew_point_min', 
                    'dew_point_max',
                    'humidity_avg',
                    'humidity_min',
                    'humidity_max',
                    'pressure_avg', 
                    'pressure_min', 
                    'pressure_max',
                    # 'uv_index_avg', 
                    # 'uv_index_min', 
                    # 'uv_index_max',
                    'precipitation_probability_avg',
                    'precipitation_probability_min',
                    'precipitation_probability_max',
                    'precipitation_intensity_avg',
                    'precipitation_intensity_min',
                    'precipitation_intensity_max'
                ],
                max_average=10,
                max_shift=10,
                ml_methods={
                    'reg': filter_items([
                        'LinearRegression',             # Linear Regression
                        # 'DecisionTreeRegressor',        # Decision Tree Regressor
                        # 'RandomForestRegressor',        # Random Forest Regressor
                        # 'GradientBoostingRegressor',    # Gradient Boosting Regressor
                        # 'PLSRegression',                # Partial Least Squares Regression
                        # 'ExtraTreeRegressor',           # Extremely Randomized Tree Regressor
                        # 'SVR',                          # Epsilon-Support Vector Regression
                        # 'MLPRegressor',                 # Multi-Layer Perceptron Regressor
                        # 'KNeighborsRegressor',          # k-Nearest Neighbors Regressor
                        # 'RegressionHoeffdingTree',      # Hoeffding Tree Regressor
                        # 'RegressionHAT',                # Hoeffding Adaptive Tree Regressor
                        # 'BaggingRegressionHT',          # Hoeffding Tree Bagging Regressor
                        # 'BaggingRegressionHAT',         # Hoeffding Adaptive Tree Bagging Regressor
                    ], ML_METHODS_REG),
                    'cls': filter_items([
                        'LogisticRegression',           # Logistic Regression
                        # 'DecisionTreeClassifier',       # Decision Tree Classifier
                        # 'ExtraTreeClassifier',          # Extremely Randomized Tree Classifier
                        # 'RandomForestClassifier',       # Random Forest Classifier
                        # 'SVC',                          # C-Support Vector Classification
                        # 'KNeighborsClassifier',         # k-Nearest Neighbors Classifier
                        # 'Perceptron',                   # Perceptron Classifier
                        # 'GaussianNB',                   # Gaussian Naïve Bayes Classifier
                        # 'HoeffdingTree',                # Hoeffding Tree Classifier
                        # 'HAT',                          # Hoeffding Adaptive Tree Classifier
                    ], ML_METHODS_CLS),
                },
                eval_methods={
                    'reg': filter_items([
                        'R2',
                        'MSE',
                    ], EVAL_METHODS_REG),
                    'cls': filter_items([
                        'F1',
                    ], EVAL_METHODS_CLS),
                },
                fs_method=FS_METHODS['kbest'],
                hpo_method=HPO_METHODS['random'],
                cv_method=CV_METHODS['time_series'],
                disc_method=DISC_METHODS['kbins'],
                horizon=[3],
                random_seed=RANDOM_SEED
            ),
        ),
    ]

    for experiment in experiments:
        experiment.run()


if __name__ == '__main__':
    main()
