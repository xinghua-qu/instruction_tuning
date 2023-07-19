import yaml
import argparse
from omegaconf import OmegaConf, DictConfig
## This function is checked to work well on date: 19 July 2023
## author: Xinghua Qu (quxinghua17@gmail.com)

def merge_configs(yaml_config, args):
    # Get the YAML configurations
    yaml_data = {}
    with open(yaml_config, 'r') as file:
        yaml_data = yaml.safe_load(file)

    # Merge argparse values into the YAML config
    for key, value in args.items():
        keys = key.split('.')
        current_dict = yaml_data
        for k in keys[:-1]:
            current_dict = current_dict.setdefault(k, {})
        current_dict[keys[-1]] = value
    return yaml_data

def parse_args():
    parser = argparse.ArgumentParser(description="Merge YAML config with command line arguments.")
    parser.add_argument('-v', '--verbose', action=CollectAllArgumentsAction)
    parser.add_argument("-c", "--config", type=str, help="YAML config file path.")
    args = parser.parse_args()
    return args

def get_config():
    args = parse_args()
    yaml_f = args.config
    args.verbose = dict({args.verbose[i].replace('--', ''): args.verbose[i + 1] for i in range(0, len(args.verbose), 2)})
    merged_config = merge_configs(args.config, args.verbose)
    # Convert the dictionary to DotDictConfig
    out_config = DotDictConfig(merged_config)
    return out_config

class CollectAllArgumentsAction(argparse.Action):
    def __init__(self, option_strings, dest, **kwargs):
        super(CollectAllArgumentsAction, self).__init__(option_strings, dest, nargs=argparse.REMAINDER, **kwargs)

    def __call__(self, parser, namespace, values, option_string=None):
        setattr(namespace, self.dest, values)


class DotDictConfig(DictConfig):
    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError:
            raise AttributeError(name)