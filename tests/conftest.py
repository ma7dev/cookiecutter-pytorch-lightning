# TODO:
import os
import sys
sys.path.insert(0, os.path.abspath(f"../project_x"))
from typing import Dict, Any
import os
import yaml
import pytest
import torch
from torch import nn
from torch.utils import data
from project_x.data.datasets.dataset import MOT
from project_x.data.samplers.sampler import BatchSampler
from project_x.detection.models.faster_rcnn import faster_rcnn
from project_x.tracking.models.lstm import LSTM
from project_x.joint.model import Joint
import project_x.utils.transforms as T
from project_x.utils.utils import collate_fn
import numpy as np
from visdom import Visdom

@pytest.fixture(scope="session")
def settings() -> Dict[str, str]:
    """[summary]
    Returns:
        Dict[str, str]: [description]
    """
    return {
        "job": "tracking_bce",
        "logger": "/home/alotaima/Logs/testing",
        "gpus": "1"
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
        "ROOT_PATH": config["ROOT_PATH"],
        "DATA_PATH": config["DATA_PATH"],
        "OUTPUT_PATH": config["OUTPUT_PATH"],
        "MODEL": settings["job"],
        "TARGET": config[settings["job"]]
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
    experiment = f"{setup['MODEL']}_{setup['TARGET']['BATCH_SIZE']}_"\
        f"{setup['TARGET']['LEARNING_RATE']}_{setup['TARGET']['MOMENTUM']}_"\
        f"{setup['TARGET']['WEIGHT_DECAY']}_{setup['TARGET']['TRACKING_LOSS']}_"\
        f"{setup['TARGET']['NUM_EPOCHS']}"
    output_dir = f"{setup['OUTPUT_PATH']}/models/{experiment}"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    return {
        "experiment": experiment,
        "output_dir": output_dir,
        "LOGGER": f"{setup['OUTPUT_PATH']}/logs/{experiment}"
    }


@pytest.fixture(scope="session")
def device(settings: Dict[str, Any]) -> torch.device:
    """[summary]
    Args:
        settings (Dict[str, Any]): [description]
    Returns:
        torch.device: [description]
    """
    os.environ["CUDA_VISIBLE_DEVICES"] = str(settings["gpus"])
    torch.manual_seed(1)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.fixture(scope="session")
def train() -> bool:
    """[summary]
    Returns:
        bool: [description]
    """
    return False


@pytest.fixture(scope="session")
def get_transform(train: bool) -> object:
    """[summary]
    Args:
        train (bool): [description]
    Returns:
        object: [description]
    """
    transforms = []
    transforms.append(T.ToTensor())
    # if train:
    #     transforms.append(T.RandomHorizontalFlip(0.5))
    return T.Compose(transforms)


@pytest.fixture(scope="session")
def dataset(setup: Dict[str, Any], get_transform: object) -> data.Dataset:
    """[summary]
    Args:
        setup (Dict[str, Any]): [description]
        get_transform (object): [description]
    Returns:
        data.Dataset: [description]
    """
    return MOT(
        f"{setup['DATA_PATH']}/train",
        get_transform,
        # split_seqs=TEST_SEQS,
        batch_size=setup['TARGET']['BATCH_SIZE'],
        test=True
    )


@pytest.fixture(scope="session")
def sampler(setup: Dict[str, Any], dataset: data.Dataset) -> data.Dataset:
    return BatchSampler(
        dataset,
        batch_size=setup['TARGET']['BATCH_SIZE'],
        shuffle=False
    )


@pytest.fixture(scope="session")
def data_loader(setup: Dict[str, Any], dataset: data.Dataset, sampler: data.Sampler) -> data.Dataset:
    return data.DataLoader(
        dataset,
        batch_sampler=sampler,
        num_workers=setup['TARGET']['NUM_WORKERS'],
        collate_fn=collate_fn
    )


@pytest.fixture(scope="session")
def detection_model(dataset: data.Dataset, device: torch.device) -> nn.Module:
    """[summary]
    Args:
        dataset (data.Dataset): [description]
        device (torch.device): [description]
    Returns:
        nn.Module: [description]
    """
    return faster_rcnn(dataset.num_classes).to(device)


@pytest.fixture(scope="session")
def tracking_model(setup: Dict[str, Any], device: torch.device) -> nn.Module:
    """[summary]
    Args:
        setup (Dict[str, Any]): [description]
        device (torch.device): [description]
    Returns:
        nn.Module: [description]
    """
    return LSTM(setup['TARGET']['BATCH_SIZE'], device).to(device)


@pytest.fixture(scope="session")
def joint(setup: Dict[str, Any], detection_model: nn.Module,
          tracking_model: nn.Module, device: torch.device) -> Dict[str, Any]:
    """[summary]
    Args:
        setup (Dict[str, Any]): [description]
        detection_model (nn.Module): [description]
        tracking_model (nn.Module): [description]
        device (torch.device): [description]
    Returns:
        Dict[str, Any]: [description]
    """
    model = Joint(detection_model, tracking_model,
                  setup['TARGET']['BATCH_SIZE'], setup['TARGET']['TRACKING_LOSS'],
                  device).to(device)

    model.freeze(
        backbone=setup['TARGET']['FREEZE_BACKBONE'],
        detection=setup['TARGET']['FREEZE_DETECTION'],
        tracking=setup['TARGET']['FREEZE_TRACKING']
    )

    # config model
    params = model.params(setup['TARGET']['LEARNING_RATE'])
    optimizer = torch.optim.SGD(
        params,
        lr=setup['TARGET']['LEARNING_RATE'],
        momentum=setup['TARGET']['MOMENTUM'],
        weight_decay=setup['TARGET']['WEIGHT_DECAY']
    )

    lr_scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=setup['TARGET']['STEP_SIZE'], gamma=setup['TARGET']['GAMMA']
    )
    return {
        "model": model,
        "optimizer": optimizer,
        "lr_scheduler": lr_scheduler
    }