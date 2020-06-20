"""
Used for shorthand setup functions to get shaper, doata, and config files
"""
from typing import Tuple, Any

from logger import setup_logging
from utils.configs import BaseConf
from utils.preprocessing import Shaper
import logging as log
import os
import numpy as np


def setup(data_sub_path: str) -> Tuple[BaseConf, Shaper, Any]:
    """
    gets data conf and sets up logging

    :param data_sub_path: string describing path to data
    :return: tuple of  conf, shaper, sparse_crimes
    """

    conf = BaseConf()

    conf.model_name = f"Mutual Info"

    conf.data_path = f"./data/processed/{data_sub_path}/"

    if not os.path.exists(conf.data_path):
        raise Exception(f"Directory ({conf.data_path}) needs to exist.")

    conf.model_path = f"{conf.data_path}models/{conf.model_name}/"
    os.makedirs(conf.data_path, exist_ok=True)
    os.makedirs(conf.model_path, exist_ok=True)

    setup_logging(save_dir=conf.model_path,
                  log_config='./logger/standard_logger_config.json',
                  default_level=log.INFO)

    with np.load(conf.data_path + "generated_data.npz") as zip_file:  # context helper ensures zip_file is closed
        if conf.use_crime_types:
            sparse_crimes = zip_file["crime_types_grids"]
        else:
            sparse_crimes = zip_file["crime_grids"]

    shaper = Shaper(data=sparse_crimes,
                    conf=conf)

    return conf, shaper, sparse_crimes
