# {{cookiecutter.project_name}}

## Setup

- Install [Conda (miniconda)](https://conda.io/miniconda.html) & [Poetry](https://python-poetry.org/docs/getting-started/installation/):
- Build and activate environment:
```bash
conda env create -f environment.yml
source activate {{cookiecutter.project_slug}}
```
- Install packages:
```bash
poetry install
```

## Commands

- Train model
```bash
poetry run train
```
- Test model
```bash
poetry run test
```
- Run pytest
```bash
poetry run pytest tests/
```

## Instructions

- To install/uninstall packages and other commands, please refer to [Poetry's documentation](https://python-poetry.org/docs/cli/)
- To run tests, please refer to [pytest's documentation](https://docs.pytest.org/en/latest/)
- To add more experiments, please refer to `config/experiments.yml`
- To run multiple experiments, please refer to `scripts/experiments.py`