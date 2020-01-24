# Global
import json
import os

# Local
from .log import Log as log


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
        disc_method,
        horizon,
        random_seed
    ):
        self.config = {
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
            'disc_method': disc_method,
            'horizon': horizon,
            'random_seed': random_seed,
        }


    def save(self, path):
        file_path = os.path.join(path, 'config.json')
        log.info(f'Saving configuration: {file_path}')
        with open(file_path, 'w') as file:
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
