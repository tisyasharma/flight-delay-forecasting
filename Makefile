.PHONY: setup features train walk-forward forecast lint

setup:
	pip install -e ".[dev,dl,track]"

features:
	python -m src.build_features

train:
	python -m src.training.train.train_lightgbm
	python -m src.training.train.train_xgboost

walk-forward:
	python -m src.training.walk_forward

forecast:
	python -m src.training.generate_forecasts

lint:
	ruff check src
