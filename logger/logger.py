import logging
import logging.config
from pathlib import Path
from pprint import pprint

from utils import read_json


def setup_logging(save_dir, file_name=None, log_config='./logger/standard_logger_config.json', default_level=None):
    """
    Setup logging configuration
    """
    log_config = Path(log_config)
    if log_config.is_file():
        config = read_json(log_config)
        if default_level:
            config["root"]["level"] = default_level
        # modify logging paths based on run config
        for _, handler in config['handlers'].items():
            if 'filename' in handler:
                if file_name:
                    handler['filename'] = f"{save_dir}/{file_name}"
                else:
                    handler['filename'] = f"{save_dir}/{handler['filename']}"

        logging.config.dictConfig(config)
    else:
        print("Warning: logging configuration file is not found in {}.".format(log_config))
        raise FileNotFoundError
