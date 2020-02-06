# Common
import argparse
import os
import json

# Local
from lib.log import Log as log
from lib.config import Config
from lib.experiment import Experiment
from lib.utils import (
    remove_comments,
    remove_trailing_commas
)

# Plotting
from pandas.plotting import register_matplotlib_converters



def parse_config(file_path):
    config = None
    if os.path.isfile(file_path):
        with open(file_path, 'r', encoding='utf-8') as json_file:
            dirty_json = json_file.read()
            config = json.loads(remove_trailing_commas(remove_comments(dirty_json)))

    return config


def parse_cli_args():
    parser = argparse.ArgumentParser(description='Run experiments with given configuratoins.')
    parser.add_argument('config', type=str, help='Configuration file in JSON format.')

    return parser.parse_args()


def main():
    args = parse_cli_args()
    config = parse_config(args.config)
    
    if not config:
        print(f'Failed to load config from: {args.config}')
        return

    register_matplotlib_converters()

    for experiment_config in config:
        name = experiment_config.get('name')
        if name:
            del experiment_config['name']
        
        description = experiment_config.get('description')
        if description:
            del experiment_config['description']
        
        Experiment(
            name=name,
            description=description,
            config=Config(**experiment_config)
        ).run()


if __name__ == '__main__':
    main()
