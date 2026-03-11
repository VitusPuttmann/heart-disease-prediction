# Predicting Heart Disease

This repository contains a machine learning project for the Kaggle competition
Predicting Heart Disease - Playground Series: Season 6 Episode 2.

The objective is to predict the probability of heart disease from clinical and
demographic patient data using a configurable preprocessing and model-training
pipeline.

Competition page:
https://www.kaggle.com/competitions/playground-series-s6e2/overview

## Project overview

The training pipeline in this project supports:

- Config-driven feature inclusion and transformation
- Cross-validation-based model evaluation
- Multiple model families under a shared registry
- Multi-model ensembling via out-of-fold predictions
- Export of trained models and prediction files

Main execution entry point:

- run.py

Core pipeline modules:

- src/loading.py
- src/preparation.py
- src/features.py
- src/cross_validation.py
- src/training.py
- src/ensemble.py

## Models included

The model registry is defined in src/models/registry.py and currently includes:

- LightGBM in several variants
- XGBboost
- CatBoost
- Logistic Regression
- Neural Net
- RealMLP

Model hyperparameters are maintained in:

- params/lightgbm.yaml
- params/xgboost.yaml
- params/catboost.yaml
- params/logistic_regression.yaml
- params/neural_net.yaml
- params/real_mlp.yaml

## Feature space

Feature definitions and metadata are maintained in configs/features.yaml.

At a high level, the project uses these feature groups:

- Original clinical and demographic variables
- Engineered cardiovascular features (for example heart-rate and risk proxies)
- Selected pairwise interaction terms
- Optional polynomial and spline expansions
- Optional additional feature families configured in run settings

Preprocessing options include one-hot encoding, winsorization,
Yeo-Johnson transformation, and standardization.

## Best-performing solution

Best run: models/boost_ensemble_2026-02-25_01

This solution is a boosting ensemble using three base models:

- LightGBM
- XGBoost
- CatBoost

It uses a broad but structured feature set composed of:

- Original clinical/demographic predictors
- Engineered heart-rate and risk summary features
- A selected subset of interaction terms
- Limited nonlinearity (targeted polynomial terms, no spline expansion)
- Targeted preprocessing transforms (winsorization and Yeo-Johnson on selected
	variables, plus standardization)

The final ensemble prediction is formed from the combined base-model
probabilities produced by the pipeline.

## Running experiments

1. Configure the run in configs/run.yaml.
2. Adjust model parameters in params/*.yaml as needed.
3. Run the pipeline:

	 python run.py

Generated artifacts are written to:

- output/<timestamp>/ for run outputs
- models/<storage_name>/ when model storage is enabled

## Repository layout

- configs/: project and run configuration files
- params/: model-specific hyperparameters
- src/: pipeline and model implementation code
- models/: archived model snapshots and stored artifacts
- submissions/: archived submission bundles
- tests/: test suite
