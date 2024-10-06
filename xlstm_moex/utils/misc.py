import yaml

from typing import Any, Dict

def load_experiment_config(config_filename: str) -> Dict[str, Any]:
    """Load .yaml file containing experiment config."""
    with open(config_filename, 'r') as stream:
        cfg = yaml.load(stream, Loader=yaml.FullLoader)
    return cfg