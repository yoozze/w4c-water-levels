# Global
import os
import re
import time
import json
from datetime import datetime
from .log import Log as log
from .utils import (
    get_scripts_path
)

# Local
from .config import Config
from .data import Data


class Experiment():
    @staticmethod
    def get_feature_filter(horizon):
        # TODO: check!
        def filter(column_name):
            match = re.match(r'^level_diff_shift_(\d+)d$', column_name)
            if match:
                if int(match.group(1)) >= horizon:
                    return True
                else:
                    return False
            match = re.match(r'^level_diff_shift_(\d+)d_average_(\d+)d$', column_name)
            if match:
                if int(match.group(1)) - int(match.group(2)) >= horizon:
                    return True
                else:
                    return False
            return True

        return filter


    def __init__(self, name, description, config):
        self.name = name
        self.description = description
        self.config = config
        self.data = Data(self.config)
        self.results = None


    def start(self):
        # Create experiment timestamp.
        self.timestamp = datetime.timestamp(datetime.now())

        # Init logging session.
        log.set(datetime.fromtimestamp(self.timestamp))

        # Create results directory
        dir_name = f'{datetime.fromtimestamp(self.timestamp)}'.replace(':', '.').replace(' ', '_')
        self.dir = get_scripts_path('experiments', dir_name)

        if not os.path.exists(self.dir):
            os.makedirs(self.dir)

        # Save configuration.
        self.config.save(self.dir)

        # Init results.
        self.results = {
            'time': time.time()
        }


    def finish(self, success=False):
        self.results['time'] = time.time() - self.results['time']
        log.info(f'Experiment finished after {self.results["time"]} seconds.')

        # Save log.
        log.save(os.path.join(self.dir, 'experiment.log'))

        # Save results.
        file_path = os.path.join(self.dir, 'results.json')
        log.info(f'Saving results: {file_path}')
        with open(file_path, 'w') as file:
            json.dump(self.results, file)


    def model(self):
        sensors = list(self.data.get_sensors().items())
        if not sensors:
            return

        horizons = self.config.get('horizon')
        ml_methods = self.config.get('ml_methods')
        ml_types = list(ml_methods.keys())
        total_count = len(sensors) * len(horizons) * sum(map(lambda t: len(ml_methods[t]), ml_types))
        count = 0

        # 1. For each sensor:
        # ===================
        self.results['sensors'] = {}
        for id, props in self.data.get_sensors().items():
            self.results['sensors'][id] = {}
            sensor_results = self.results['sensors'][id]

            # 2. For regression/classification:
            # =================================
            sensor_results['types'] = {}
            for ml_type in ml_types:
                sensor_results['types'][ml_type] = {}
                type_results = sensor_results['types'][ml_type]

                # 3. For each prediction horizon:
                # ===============================
                type_results['horizons'] = {}
                for horizon in horizons:
                    type_results['horizons'][horizon] = {}
                    horizon_results = type_results['horizons'][horizon]

                    # Prepare data.
                    dataset = self.data.get(id, column_filter=Experiment.get_feature_filter(horizon), ml_type=ml_type)
                    df_X = dataset.iloc[:, 2:]
                    df_y_abs = dataset['level']
                    df_y_rel = dataset['level_diff']

                    X = None
                    y = df_y_rel.values.astype(float)
                    ya = df_y_abs.values.astype(float)

                    log.info(f'Data prepared: {dataset.shape}')

                    # Feature selection (model agnostic)
                    # ----------------------------------

                    fs_method = self.config.get('fs_method')
                    fs_params = fs_method.get('params', {})
                    fs_has_estimator = 'estimator' in fs_params
                    fs_estimator_is_defined = fs_has_estimator and fs_params.get('estimator') != None
                    if not fs_has_estimator or fs_estimator_is_defined:
                        log.info(f'Selecting model agnostic features with {fs_method["name"]} ...')
                        
                        # Prepare feature selection model.
                        fs_model = fs_method['class'](**fs_params)
                        
                        # Train FS model.
                        t = time.time()
                        X = fs_model.fit_transform(df_X.values.astype(float), y)
                        t = time.time() - t
                        
                        # Store results.
                        selected_features = list(df_X.columns[fs_model.get_support(indices=True)])
                        horizon_results['feature_selection'] = {
                            'selected': selected_features,
                            'time': t
                        }

                        log.info(f'Selected {len(selected_features)} fetures.')

                    # 4. For each ML method:
                    # ======================
                    horizon_results['methods'] = {}
                    for ml_method in ml_methods[ml_type]:
                        horizon_results['methods'][ml_method['name']] = {}
                        method_results = horizon_results['methods'][ml_method['name']]

                        count += 1
                        log.info(f'Building {ml_type.upper()} model {count}/{total_count} for sensor {id} \"{props["name"]}, {props["waterbody_name"]}\" with [h={horizon}] {ml_method["name"]}')
                        
                        ml_model = ml_method['class'](**ml_method.get('params', {}))

                        # Feature selection (model specific)
                        # ----------------------------------

                        if fs_has_estimator and not fs_estimator_is_defined:
                            log.info(f'Selecting features for {ml_method["name"]} with {fs_method["name"]} ...')

                            # Prepare feature selection model.
                            fs_params_new = fs_params.copy()
                            fs_params_new['estimator'] = ml_model
                            fs_model = fs_method['class'](**fs_params_new)
                            
                            # Train FS model.
                            t = time.time()
                            X = fs_model.fit_transform(df_X.values.astype(float), y)
                            t = time.time() - t
                            
                            # Store results.
                            selected_features = list(df_X.columns[fs_model.get_support(indices=True)])
                            method_results['feature_selection'] = {
                                'selected': selected_features,
                                'time': t
                            }

                            log.info(f'Selected {len(selected_features)} fetures.')

                        # Feature scaling
                        # ---------------

                        fsc_method = ml_method.get('scaler')
                        if fsc_method:
                            log.info(f'Scaling features ...')

                            # Prepare feature scaling model.
                            fsc_model = fsc_method['class'](**fsc_method.get('params', {}))

                            # Train FSC model.
                            t = time.time()
                            X = fsc_model.fit_transform(X)
                            t = time.time() - t

                            # Store results.
                            method_results['feature_scaling'] = {
                                'time': t
                            }

                        # Hyperparameter optimization
                        # ---------------------------
                        
                        # ml_params_grid = ml_method.get('params_grid', {})
                        # if len(ml_params_grid.keys()):
                        #     log.info(f'Optimizing hyperparameters ...')

                        #     # Prepare hyperparameter optimization model.
                        #     hpo_method = self.config.get('hpo_method')
                        #     hpo_params = hpo_method.get('params', {}).copy()
                        #     hpo_params['estimator'] = ml_model

                        #     for key in ['param_grid', 'param_distributions']:
                        #         if key in hpo_params:
                        #             hpo_params[key] = ml_params_grid
                        #             break
                        #     hpo_model = hpo_method['class'](**hpo_params)

                        #     # Train HPO model.
                        #     t = time.time()
                        #     hpo_model.fit(X, y)
                        #     t = time.time() - t

                        #     # Store results.
                        #     method_results['hyperparameter_optimization'] = {
                        #         'selected': hpo_model.best_params_,
                        #         'time': t
                        #     }

                        #     # Prepare the model with optimized parameters.
                        #     ml_model = ml_method['class'](**hpo_model.best_params_)

                        #     log.info(f'Hyperparameters optimized: {hpo_model.best_params_}')

                        # Cross-Validation
                        # ----------------
                        
                        cv_method = self.config.get('cv_method')
                        cv_params = cv_method.get('params', {}).copy()
                        if cv_params['n_splits'] == None:
                            # cv_params['n_splits'] = min(5, round(len(y) / 365))
                            cv_params['n_splits'] = round(len(y) / 365)
                        cv_model = cv_method['class'](**cv_params)
                        i = 0
                        method_results['cv'] = {}
                        for train_index, test_index in cv_model.split(X):
                            i += 1

                            # Skip first iteration.
                            if cv_method['name'] in ['time_series'] and i == 1:
                                continue

                            log.info(f'Cross-validating {i - 1}/{cv_params["n_splits"] - 1} ...')

                            method_results['cv'][i - 1] = {}
                            cv_results = method_results['cv'][i - 1]

                            # Split training set to two sets, training and validation.
                            test_len = len(test_index)
                            validate_index = train_index[-test_len:]
                            train_index = train_index[:-test_len]

                            X_train, X_validate, X_test = X[train_index], X[validate_index], X[test_index]
                            y_train, y_validate, y_test, ya_test = y[train_index], y[validate_index], y[test_index], ya[test_index]

                            # Hyperparameter optimization
                            # ---------------------------
                        
                            ml_params_grid = ml_method.get('params_grid', {})
                            if len(ml_params_grid.keys()):
                                log.info(f'Optimizing hyperparameters ...')

                                # Prepare hyperparameter optimization model.
                                hpo_method = self.config.get('hpo_method')
                                hpo_params = hpo_method.get('params', {}).copy()
                                hpo_params['estimator'] = ml_model

                                for key in ['param_grid', 'param_distributions']:
                                    if key in hpo_params:
                                        hpo_params[key] = ml_params_grid
                                        break
                                hpo_model = hpo_method['class'](**hpo_params)

                                # Train HPO model.
                                t = time.time()
                                hpo_model.fit(X_validate, y_validate)
                                t = time.time() - t

                                # Store results.
                                cv_results['hyperparameter_optimization'] = {
                                    'selected': hpo_model.best_params_,
                                    'time': t
                                }

                                # Prepare the model with optimized parameters.
                                ml_model = ml_method['class'](**hpo_model.best_params_)

                                log.info(f'Hyperparameters optimized: {hpo_model.best_params_}')

                            # Modelling
                            # ---------

                            # Train.
                            tt = time.time()
                            ml_model.fit(X_train, y_train)
                            tt = time.time() - tt

                            # Predict.
                            tp = time.time()
                            y_pred = ml_model.predict(X_test)
                            tp = time.time() - tp

                            # Store results.
                            cv_results['modelling'] = {
                                'modelling_time': tt,
                                'predicting_time': tp,
                                'y_pred': y_pred.tolist(),
                                'y_test': y_test.tolist(),
                                'ya_test': ya_test.tolist()
                            }

                            # Evaluate.
                            cv_results['evaluation'] = {}
                            eval_methods = self.config.get('eval_methods')
                            for eval_method in eval_methods[ml_type]:
                                eval_params = eval_method.get('params', {})
                                cv_results['evaluation'][eval_method['name']] = eval_method['def'](y_pred, y_test, **eval_params)

                            if ml_type == 'cls':
                                # Convert discrete values (classes) back to real values.
                                y_pred_reg = self.data.undiscretize_array(y_pred)
                                y_test_reg = self.data.undiscretize_array(y_test)

                                for eval_method in eval_methods['reg']:
                                    eval_params = eval_method.get('params', {})
                                    cv_results['evaluation'][eval_method['name']] = eval_method['def'](y_pred_reg, y_test_reg, **eval_params)
        # end loop


    def run(self):
        self.start()

        date_time = f'{datetime.fromtimestamp(round(self.timestamp))}'.replace(' ', ' at ')
        header = f'* Starting new experiment on {date_time} *'
        len_header = len(header)
        log.print()
        log.print('*' * len_header)
        log.print(header)
        log.print('*' * len_header)
        log.print()
        log.print(f'Experiment: {self.name}')
        log.print(f'Description: {self.description}')
        log.print()

        self.model()

        self.finish()


def main():
    pass


if __name__ == '__main__':
    main()
