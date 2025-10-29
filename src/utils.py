import os
import sys
import yaml

import pandas as pd

__root__ = os.getcwd()
sys.path.insert(0, __root__)

def load_config() -> dict:
    CONFIG_PATH = os.path.join(__root__, "config.yaml")
    with open(CONFIG_PATH, "r") as config_file:
        config = yaml.safe_load(config_file)
    return config


def read_file(filepath: str):
    """
    Reads a file and returns its contents. Supports CSV and TXT formats.
    Args:
        filepath (str): The path to the file to be read.
    Returns:
        pd.DataFrame or list: The contents of the file as a DataFrame (for CSV) or a list (for TXT).
    """
    if filepath.endswith(".csv"):
        df = pd.read_csv(filepath)
        return df
    elif filepath.endswith(".txt"):
        with open(filepath, "r") as file:
            content = file.readlines()
        return [line.strip() for line in content]
    else:
        raise ValueError("Unsupported file format.")