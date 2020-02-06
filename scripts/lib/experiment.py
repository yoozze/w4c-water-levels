# Common
import copy
import os
import re
import time
import json
import traceback
from datetime import datetime

# Local
from .log import Log as log
from .utils import (
    # call,
    get_scripts_path
)
from .config import (
    call,
    Config
)
from .data import Data



class Experiment():
    @staticmethod
    def get_feature_filter(horizon):
        # TODO: check!
        def filter(column_name):
            match = re.match(r'^level_diff_shift_(\d+)d', column_name)
            if match:
                if int(match.group(1)) >= horizon:
                    return True
                else:
                    return False
            # match = re.match(r'^level_diff_shift_(\d+)d_average_(\d+)d$', column_name)
            # if match:
            #     if int(match.group(1)) >= horizon:
            #         return True
            #     else:
            #         return False
            return True

        return filter


    def __init__(self, name, description, config):
        self.name = name
        self.description = description
        self.config = config
        self.data = Data(self.config)
        self.results = None


    def print_header(self):
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


    def start(self):
        # Create experiment timestamp.
        self.timestamp = datetime.timestamp(datetime.now())

        # Init logging session.
        log.set(datetime.fromtimestamp(self.timestamp))

        # Create results directory.
        dir_name = f'{datetime.fromtimestamp(self.timestamp)}'.replace(':', '.').replace(' ', '_')
        self.dir = get_scripts_path('experiments', dir_name)

        if not os.path.exists(self.dir):
            os.makedirs(self.dir)

        # Print header to stdout.
        self.print_header()

        # Save configuration.
        self.config.save(self.dir)

        # Init results.
        self.results = {
            'time': time.time()
        }


    def finish(self, code=0):
        self.results['time'] = time.time() - self.results['time']
        if code:
            log.error(f'Experiment failed after {self.results["time"]} seconds.')
        else:
            log.info(f'Experiment finished after {self.results["time"]} seconds.')

        # Save results.
        file_path = os.path.join(self.dir, 'results.json')
        log.info(f'Saving results: {file_path}')
        with open(file_path, 'w', encoding="utf-8") as file:
            json.dump(self.results, file)

        # Save log.
        log.save(os.path.join(self.dir, 'experiment.log'))

        # Reset stdout/stderr.
        log.reset()


    def get_ml_method_name(self, method):
        ml_method_name = method['name']
        ml_params = method.get('params', {})
        ml_estimator = ml_params.get('base_estimator')

        if ml_estimator:
            ml_method_name += f':{ml_estimator["name"]}'

        return ml_method_name


    def init_ml_model(self, method):
        ml_params = copy.deepcopy(method.get('params', {}))
        ml_estimator = ml_params.get('base_estimator')

        if ml_estimator:
            ml_params['base_estimator'] = call(ml_estimator['name'], ml_estimator.get('params', {}))

        return call(method['name'], ml_params)


    def select_features(self, method, df_X, y, estimator=None):
        if not method:
            return None

        fs_params = copy.deepcopy(method.get('params', {}))
        fs_estimator = fs_params.get('estimator')
        
        # Prepare feature selection model.
        if estimator != None:
            # Use provided estimator.
            log.info(f'Selecting features for {type(estimator).__name__} with {method["name"]} ...')
            fs_params['estimator'] = estimator
        else:
            if fs_estimator:
                # Use predefined estimator.
                log.info(f'Selecting features for predefined estimator {fs_estimator["name"]} with {method["name"]} ...')
                fs_params['estimator'] = call(fs_estimator['name'], fs_estimator.get('params', {}))
            elif 'estimator' in fs_params:
                # Estimator should be provided.
                return None
            else:
                # Feature selection method doesn't need estimator.
                log.info(f'Selecting features with {method["name"]} ...')

        fs_model = call(method['name'], fs_params)
        
        # Train FS model.
        t = time.time()
        # X = fs_model.fit_transform(df_X.values.astype(float), y)
        fs_model.fit(df_X.values.astype(float), y)
        t = time.time() - t
        
        # Get list of selected feature names.
        selected_features = []
        if method['name'] == 'SelectKBest':
            features_len = len(self.config.get('features'))
            # features = list(df_X.columns[:features_len + 3])
            features = list(df_X.columns[features_len:features_len + 3])
            for i, feature in enumerate(features):
                if fs_model.scores_[i] > 0 and feature not in selected_features:
                    selected_features.append(feature)

            feature_indices = fs_model.scores_.argsort()[::-1]
            k = fs_params.get('k', 20)
            i = 0
            while len(selected_features) < k and i < len(feature_indices) and fs_model.scores_[feature_indices[i]] > 0:
                feature = df_X.columns[feature_indices[i]]
                if feature not in selected_features:
                    selected_features.append(feature)
                i += 1
        else:
            # selected_features = list(df_X.columns[fs_model._get_support_mask()])
            selected_features = list(df_X.columns[fs_model.support_])

        X = df_X[selected_features].values.astype(float)

        # Prepare report.
        report = {
            'selected': selected_features,
            'time': t
        }

        log.info(f'Selected {len(selected_features)} fetures.')

        return X, report


    def scale_features(self, method, X):
        if not method:
            return None

        log.info(f'Scaling features ...')

        # Prepare feature scaling model.
        fsc_model = call(method['name'], method.get('params', {}))

        # Train FSC model.
        t = time.time()
        X = fsc_model.fit_transform(X)
        t = time.time() - t

        # prepare report.
        report = {
            'time': t
        }

        return X, report


    def optimize_hyperparameters(self, method, model, params_grid, X, y):
        method = self.config.get('hpo_method')

        if not method:
            return None

        log.info(f'Optimizing hyperparameters ...')

        # Prepare hyperparameter optimization model.
        hpo_params = copy.deepcopy(method.get('params', {}))
        hpo_params['estimator'] = model

        for key in ['param_grid', 'param_distributions']:
            if key in hpo_params:
                hpo_params[key] = params_grid
                break

        hpo_model = call(method['name'], hpo_params)

        # Train HPO model.
        t = time.time()
        hpo_model.fit(X, y)
        t = time.time() - t

        # Prepare report.
        report = {
            'selected': hpo_model.best_params_,
            'time': t
        }

        log.info(f'Hyperparameters optimized: {hpo_model.best_params_}')

        return report


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
        for sensor_id, sensor_info in self.data.get_sensors().items():
            self.results['sensors'][sensor_id] = {}
            sensor_results = self.results['sensors'][sensor_id]

            # Skip sensor with missing data.
            self.data.get(sensor_id, raw=True)
            sensor_results['missing_values'] = sensor_info['missing_values']
            missing_values_sum = sum(list(sensor_info['missing_values'].values()))
            if missing_values_sum:
                log.warning('Missing sensor data:', list(sensor_info['missing_values'].values()))
                log.warning(f'Skipping sensor {sensor_id}.')
                continue

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
                    dataset = self.data.get(sensor_id, column_filter=Experiment.get_feature_filter(horizon), ml_type=ml_type)
                    
                    if ml_type == 'cls' and not type_results.get('bins'):
                        type_results['bins'] = sensor_info['bins']

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
                    if fs_method:
                        fs_result = self.select_features(fs_method, df_X, y)
                        
                        if fs_result:
                            X, fs_report = fs_result
                            horizon_results['feature_selection'] = fs_report

                    # 4. For each ML method:
                    # ======================
                    horizon_results['methods'] = {}
                    for ml_method in ml_methods[ml_type]:
                        ml_method_name = self.get_ml_method_name(ml_method)
                        horizon_results['methods'][ml_method_name] = {}
                        method_results = horizon_results['methods'][ml_method_name]

                        count += 1
                        log.info(f'Building {ml_type.upper()} model {count}/{total_count} for sensor {sensor_id} \"{sensor_info["name"]}, {sensor_info["waterbody_name"]}\" with [h={horizon}] {ml_method_name}')
                        
                        ml_model = self.init_ml_model(ml_method)

                        # Feature selection (model specific)
                        # ----------------------------------

                        if fs_method and 'feature_selection' not in horizon_results:
                            fs_result = self.select_features(fs_method, df_X, y, ml_model)
                            
                            if fs_result:
                                X, fs_report = fs_result
                                method_results['feature_selection'] = fs_report

                        # Feature scaling
                        # ---------------

                        fsc_method = ml_method.get('scaler')
                        if fsc_method:
                            fsc_result = self.scale_features(fsc_method, X)
                            
                            if fsc_result:
                                X, fsc_report = fsc_result
                                method_results['feature_scaling'] = fsc_report

                        # Cross-Validation
                        # ----------------
                        
                        cv_method = self.config.get('cv_method')
                        cv_params = copy.deepcopy(cv_method.get('params', {}))
                        if cv_params['n_splits'] == None:
                            cv_params['n_splits'] = round(len(y) / 365)
                        cv_model = call(cv_method['name'], cv_params)
                        
                        skip_first_n = 0
                        if cv_method['name'] in ['TimeSeriesSplit']:
                            skip_first_n = 5

                        i = 0
                        method_results['cv'] = {}
                        for train_index, test_index in cv_model.split(X):
                            i += 1

                            # Skip first iteration.
                            if skip_first_n and i <= skip_first_n:
                                continue
                            
                            cv_k = cv_params['n_splits'] - skip_first_n
                            cv_ki = i - skip_first_n
                            log.info(f'Cross-validating {cv_ki}/{cv_k} ...')

                            method_results['cv'][cv_ki] = {}
                            cv_results = method_results['cv'][cv_ki]

                            # Split training set to two sets, training and validation.
                            test_len = len(test_index)
                            validate_index = train_index[-test_len:]
                            train_index = train_index[:-test_len]

                            X_train, X_validate, X_test = X[train_index], X[validate_index], X[test_index]
                            y_train, y_validate, y_test, ya_test = y[train_index], y[validate_index], y[test_index], ya[test_index]

                            # Hyperparameter optimization
                            # ---------------------------
                        
                            hpo_method = self.config.get('hpo_method')
                            ml_params_grid = ml_method.get('params_grid', {})

                            if hpo_method and len(ml_params_grid.keys()):
                                hpo_report = self.optimize_hyperparameters(hpo_method, ml_model, ml_params_grid, X_validate, y_validate)

                                if hpo_report:
                                    opt_ml_params = hpo_report['selected']
                                    new_ml_method = copy.deepcopy(ml_method)
                                    new_ml_params = new_ml_method.get('params', {})
                                    new_ml_method['params'] = { **new_ml_params, **opt_ml_params }
                                    ml_model = self.init_ml_model(new_ml_method)
                                    cv_results['hyperparameter_optimization'] = hpo_report

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
                                cv_results['evaluation'][eval_method['name']] = call(eval_method['name'], { 'y_true': y_test, 'y_pred': y_pred, **eval_params })

                            if ml_type == 'cls':
                                # Convert discrete values (classes) back to real values.
                                y_pred_reg = self.data.undiscretize_array(y_pred)
                                y_test_reg = self.data.undiscretize_array(y_test)

                                for eval_method in eval_methods['reg']:
                                    eval_params = eval_method.get('params', {})
                                    cv_results['evaluation'][eval_method['name']] = call(eval_method['name'], { 'y_true': y_test_reg, 'y_pred': y_pred_reg, **eval_params })

                            log.info('Cross-validation results:', ', '.join([f'{t[0]} = {t[1]}' for t in cv_results['evaluation'].items()]))

                            # Print average evaluation results.
                            if cv_ki == cv_k and cv_k > 0:
                                avg = {}
                                for k, v in method_results['cv'].items():
                                    if not avg:
                                        avg = {k1: float(v1) / float(cv_k) for k1, v1 in v['evaluation'].items()}
                                    else:
                                        for k1, v1 in v['evaluation'].items():
                                            avg[k1] += float(v1) / float(cv_k)
                                log.info('Cross-validation average:', ', '.join([f'{t[0]} = {t[1]}' for t in avg.items()]))


        # end loop


    def run(self):
        self.start()

        code = 0
        try:
            self.model()
        except Exception:
            code = 1
            log.print(traceback.format_exc())

        self.finish(code)


def main():
    pass


if __name__ == '__main__':
    main()
