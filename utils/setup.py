"""
Used for shorthand setup functions to get shaper, doata, and config files
"""
import logging as log
import numpy as np
import os
from pandas import read_pickle
from typing import Tuple, Any, List

from logger import setup_logging
from utils.configs import BaseConf
from utils.preprocessing import Shaper


def setup(data_sub_path: str, model_name: str = "Analysis") -> Tuple[BaseConf, Shaper, Any, List[str]]:
    """
    gets data conf and sets up logging

    :param data_sub_path: string describing path to data
    :param model_name: model_name
    :return: tuple of  conf, shaper, sparse_crimes
    """

    conf = BaseConf()

    conf.model_name = model_name

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
        sparse_crimes = zip_file["crime_types_grids"]
        crime_feature_indices = zip_file["crime_feature_indices"]

    shaper = Shaper(data=sparse_crimes,
                    conf=conf)

    t_range = read_pickle(conf.data_path + "t_range.pkl")

    return conf, shaper, sparse_crimes, t_range, crime_feature_indices
