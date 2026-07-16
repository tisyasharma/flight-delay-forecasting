.PHONY: setup features data-checks weather refresh train train-quantile walk-forward ablation recursive-eval forecast test lint mlflow-ui

# the project venv is the supported interpreter, a conda python on PATH
# shadows it with incompatible pytest and torch versions
PYTHON ?= venv/bin/python

setup:
	python3 -m venv venv
	$(PYTHON) -m pip install -e ".[dev,track]"

features:
	$(PYTHON) -m src.build_features
	$(PYTHON) -m src.data_checks

data-checks:
	$(PYTHON) -m src.data_checks

weather:
	$(PYTHON) -m src.training.fetch_weather_data
	$(PYTHON) -m src.weather.gfs_history

# the monthly ritual: drop the new BTS month into data/raw/, then run this
refresh: weather
	$(PYTHON) -m src.process
	$(PYTHON) -m src.build_features
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

