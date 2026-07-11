.PHONY: setup features data-checks train train-quantile walk-forward ablation recursive-eval forecast test lint mlflow-ui backup-docs

# the project venv is the supported interpreter, a conda python on PATH
# shadows it with incompatible pytest and torch versions
PYTHON ?= venv/bin/python
BACKUP_DIR ?= $(HOME)/Backups/flight-delay-forecasting-docs

setup:
	$(PYTHON) -m pip install -e ".[dev,dl,track]"

features:
	$(PYTHON) -m src.build_features
	$(PYTHON) -m src.data_checks

data-checks:
	$(PYTHON) -m src.data_checks

train:
	$(PYTHON) -m src.training.train.train_lightgbm
	$(PYTHON) -m src.training.train.train_xgboost

train-quantile:
	$(PYTHON) -m src.training.train.train_lightgbm_quantile

walk-forward:
	$(PYTHON) -m src.training.walk_forward

ablation:
	$(PYTHON) -m src.training.ablation

recursive-eval:
	$(PYTHON) -m src.forecasting.recursive_eval

forecast:
	$(PYTHON) -m src.training.generate_forecasts

test:
	venv/bin/pytest

lint:
	venv/bin/ruff check src tests

mlflow-ui:
	venv/bin/mlflow ui --backend-store-uri sqlite:///mlflow.db

backup-docs:
	mkdir -p $(BACKUP_DIR)
	rsync -a docs/*.local.md $(BACKUP_DIR)/
