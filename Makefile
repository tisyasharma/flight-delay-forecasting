.PHONY: setup features train train-quantile walk-forward forecast test lint mlflow-ui

setup:
	pip install -e ".[dev,dl,track]"

features:
	python -m src.build_features

train:
	python -m src.training.train.train_lightgbm
	python -m src.training.train.train_xgboost

train-quantile:
	python -m src.training.train.train_lightgbm_quantile

walk-forward:
	python -m src.training.walk_forward

forecast:
	python -m src.training.generate_forecasts

test:
	pytest

lint:
	ruff check src tests

mlflow-ui:
	mlflow ui --backend-store-uri sqlite:///mlflow.db
