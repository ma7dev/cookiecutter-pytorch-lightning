import pytest
import numpy as np
from torch.utils.data import DataLoader
from torch.utils.data.sampler import Sampler
from data.sampler import BatchSampler
from utils.utils import collate_fn


@pytest.fixture
def data_loader(setup, dataset, sampler) -> DataLoader:
    return DataLoader(
        dataset,
        batch_sampler=sampler,
        num_workers=setup['TARGET']['NUM_WORKERS'],
        collate_fn=collate_fn,
    )

@pytest.fixture
def sampler(setup, dataset) -> Sampler:
    return BatchSampler(dataset, batch_size=setup['TARGET']['BATCH_SIZE'], shuffle=False)

def test_dataset(setup, dataset, get_transform):
    assert f"{setup['DATA_PATH']}/train" == dataset.root
    assert setup['TARGET']['BATCH_SIZE'] == dataset._batch_size
    # transformation
    img = np.random.rand(16, 16, 3)
    target = np.random.rand(1)
    desired_img, desired_target = get_transform(img, target)
    actual_img, actual_target = dataset.transforms(img, target)
    assert torch.equal(desired_img, actual_img)
    assert np.equal(desired_target, actual_target)

    # sequence length
    test_size = 20
    num_seqs = 7
    desired_total_seqs_len = test_size * num_seqs
    actual_total_seqs_len = 0
    for _, source in dataset.sources.items():
        actual_total_seqs_len += source["seq_len"]
    assert desired_total_seqs_len == actual_total_seqs_len

def test_sampler(setup, dataset, sampler):
    assert setup['TARGET']['BATCH_SIZE'] == sampler.batch_size
    assert dataset == sampler.dataset
    assert dataset.sources == sampler.sources

    # length
    desired_iter_num = 0
    for _, source in dataset.sources.items():
        possible_end = source["seq_len"] - setup['TARGET']['BATCH_SIZE']
        for _ in range(0, possible_end, setup['TARGET']['BATCH_SIZE']):
            desired_iter_num += 1
    assert desired_iter_num == sampler.iter_num