# TODO:
from typing import Dict, Any
import os
import sys
from rich import print, pretty, inspect, traceback
pretty.install()
traceback.install()
import os
import yaml
import string
import random
import pytest
import torch
import wandb
import numpy as np
from pytorch_lightning import loggers
from visdom import Visdom

def pytest_configure():
    try:
        import rich
    except ImportError:
        pass
    else:
        rich.get_console()  # this is new !
        rich.reconfigure(soft_wrap=False)

@pytest.fixture(scope="session")
def settings() -> Dict[str, str]:
    """[summary]
    Returns:
        Dict[str, str]: [description]
    """
    return {
        "config_path": "./config/experiments.yml",
        "job": "main"
    }


@pytest.fixture(scope="session")
def vis():
    port = 8097
    server = "http://localhost"
    base_url = '/'
    username = ''
    password = ''
    vis = Visdom(port=port, server=server, base_url=base_url, username=username, password=password,
                 use_incoming_socket=True, env="tests")
    assert vis.check_connection(timeout_seconds=3), 'No connection could be formed quickly'
    return vis


@pytest.fixture(scope="session")
def setup(settings: Dict[str, str]) -> Dict[str, Any]:
    """[summary]
    Args:
        settings (Dict[str, str]): [description]
    Returns:
        Dict[str, Any]: [description]
    """
    config = yaml.load(open(settings["config_path"]), Loader=yaml.FullLoader)
    return {
        "DATA_PATH": config["DATA_PATH"],
        "OUTPUT_PATH": config["OUTPUT_PATH"],
        "PROJECT": config["PROJECT"],
        "JOB": settings["job"],
        "HYP": config[settings["job"]]
    }


@pytest.fixture(scope="session")
def rgbs():
    return np.random.rand(100000, 3)


@pytest.fixture(scope="session")
def output(setup: Dict[str, Any]) -> Dict[str, Any]:
    """[summary]
    Args:
        setup (Dict[str, Any]): [description]
    Returns:
        Dict[str, Any]: [description]
    """
    random_str = ''.join(random.choices(
        string.ascii_uppercase + string.digits, k=5))
    experiment = f"{random_str}_{setup['JOB']}_{setup['HYP']['BATCH_SIZE']}_{setup['HYP']['LEARNING_RATE']}_{setup['HYP']['MAX_EPOCH']}"
    output_dir = f"{setup['OUTPUT_PATH']}/checkpoints/{experiment}"
    logger = f"{setup['OUTPUT_PATH']}/logs/{experiment}"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    return {
        "experiment": experiment,
        "output_dir": output_dir,
        "LOGGER": logger
    }


@pytest.fixture(scope="session")
def device(setup: Dict[str, Any]) -> torch.device:
    """[summary]
    Args:
        settings (Dict[str, Any]): [description]
    Returns:
        torch.device: [description]
    """
    os.environ["CUDA_VISIBLE_DEVICES"] = str(setup['HYP']["GPUS"])
    torch.manual_seed(1)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def loggers(output):
    return loggers.TensorBoardLogger(save_dir=output["LOGGER"])