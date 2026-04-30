# LightGBM Regressor Models

## Overview

This document describes two public classes used in the sales forecasting workflow:

- `LightGBMRegressorBaseLine`
- `LightGBMRegressorAnomaly`

Both classes train a `LightGBMRegressor` model and follow a very similar structure.  
The main difference is their intended use in the project:

- `LightGBMRegressorBaseLine` is the baseline regression model
- `LightGBMRegressorAnomaly` is the regression model that works with anomaly-related features

These classes are designed to support:

- preprocessing of numeric and categorical features
- KFold cross-validation
- hyperparameter tuning with `GridSearchCV`
- training with previously saved best parameters
- prediction and evaluation
- model persistence
- metadata and metrics export

The descriptions below are based on the uploaded source files fileciteturn5file0 fileciteturn5file1

---

## Common Purpose

Both classes help build a regression pipeline for aggregated sales data.

Typical use cases include:

- demand estimation
- sales value prediction
- model comparison between baseline and anomaly-enhanced approaches
- reproducible training workflows
- structured model export for later reuse

---

# Class: `LightGBMRegressorBaseLine`

## Overview

`LightGBMRegressorBaseLine` is a baseline regression model built with LightGBM.

It trains a model without anomaly detection features and is useful as a comparison point against more advanced approaches, such as a hybrid solution that uses anomaly information.

---

## Constructor

```python
LightGBMRegressorBaseLine(
    target_col: str,
    numeric_features: List[str],
    categorical_features: Optional[List[str]] = None,
    model_dir: str = "models/baseline",
    random_state: int = 42,
    n_splits: int = 3,
    scoring: str = "neg_root_mean_squared_error"
)
```

## Parameters

- `target_col` (`str`): Target column to predict.
- `numeric_features` (`List[str]`): List of numeric input features.
- `categorical_features` (`Optional[List[str]]`): List of categorical input features.
- `model_dir` (`str`): Folder used to save the model, parameters, and metrics.
- `random_state` (`int`): Random seed for reproducibility.
- `n_splits` (`int`): Number of folds used in KFold cross-validation.
- `scoring` (`str`): Scoring metric used by `GridSearchCV`.

## Main Attributes

- `pipeline`: Full training pipeline.
- `grid_search`: Fitted `GridSearchCV` object.
- `best_model`: Best trained model pipeline.
- `best_params`: Best hyperparameters found during grid search.

---

## Public Methods

### `fit(df_train: pd.DataFrame, param_grid: Optional[Dict[str, List[Any]]] = None) -> GridSearchCV`

Train the baseline model using `GridSearchCV`.

The method:

1. validates the input DataFrame
2. selects the required feature columns and target
3. builds the preprocessing and modeling pipeline
4. creates KFold cross-validation
5. runs hyperparameter tuning
6. stores the best model and best parameters

#### Parameters

- `df_train` (`pd.DataFrame`): Training DataFrame with features and target.
- `param_grid` (`Optional[Dict[str, List[Any]]]`): Custom hyperparameter grid. If not provided, the class uses its internal default grid.

#### Returns

- `GridSearchCV`: Fitted grid search object.

---

### `fit_with_best_params(df_train: pd.DataFrame, params_path: Optional[str] = None) -> Pipeline`

Train the model using previously saved best hyperparameters.

This method is useful when the hyperparameter search was already executed before and you want to avoid running it again.

#### Parameters

- `df_train` (`pd.DataFrame`): Training DataFrame with features and target.
- `params_path` (`Optional[str]`): Path to a JSON file with previously saved best parameters.

#### Returns

- `Pipeline`: Trained pipeline with the loaded best parameters.

#### Important note

This method expects `self.best_params` to be available, either from a previous `fit()` call or from a loading step.

---

### `predict(df: pd.DataFrame) -> np.ndarray`

Generate predictions with the best trained model.

#### Parameters

- `df` (`pd.DataFrame`): DataFrame containing the required input features.

#### Returns

- `np.ndarray`: Predicted values.

---

### `evaluate(df_test: pd.DataFrame) -> Dict[str, float]`

Evaluate the trained model on a test dataset.

The method calculates:

- MAE
- RMSE
- R2

#### Parameters

- `df_test` (`pd.DataFrame`): Test DataFrame with features and target.

#### Returns

- `Dict[str, float]`: Dictionary with the evaluation metrics.

---

### `save_model(filename: str = "lightgbm_baseline_model.joblib") -> Path`

Save the trained baseline model to disk.

#### Parameters

- `filename` (`str`): Output model file name.

#### Returns

- `Path`: Full path of the saved model file.

---

### `save_best_params(filename: str = "lightgbm_baseline_best_params.json") -> Path`

Save the best hyperparameters and training metadata to a JSON file.

The saved file includes:

- best parameters
- best score
- scoring metric
- number of folds
- target column
- numeric features
- categorical features

#### Parameters

- `filename` (`str`): Output JSON file name.

#### Returns

- `Path`: Full path of the saved JSON file.

---

### `save_metrics(metrics: Dict[str, float], filename: str = "lightgbm_baseline_metrics.json") -> Path`

Save evaluation metrics to a JSON file.

#### Parameters

- `metrics` (`Dict[str, float]`): Dictionary with evaluation results.
- `filename` (`str`): Output metrics file name.

#### Returns

- `Path`: Full path of the saved metrics file.

---

### `load_model(model_path: str) -> None`

Load a trained model pipeline from disk.

#### Parameters

- `model_path` (`str`): Path to the saved joblib model.

#### Returns

- `None`

---

# Class: `LightGBMRegressorAnomaly`

## Overview

`LightGBMRegressorAnomaly` is a LightGBM regression model designed to work with anomaly-related features.

It is useful for comparing a standard baseline approach against a richer feature set that includes anomaly detection signals.

---

## Constructor

```python
LightGBMRegressorAnomaly(
    target_col: str,
    numeric_features: List[str],
    categorical_features: Optional[List[str]] = None,
    model_dir: str = "models/baseline",
    random_state: int = 42,
    n_splits: int = 3,
    scoring: str = "neg_root_mean_squared_error"
)
```

## Parameters

- `target_col` (`str`): Target column to predict.
- `numeric_features` (`List[str]`): List of numeric input features.
- `categorical_features` (`Optional[List[str]]`): List of categorical input features.
- `model_dir` (`str`): Folder used to save the model, parameters, and metrics.
- `random_state` (`int`): Random seed for reproducibility.
- `n_splits` (`int`): Number of folds used in KFold cross-validation.
- `scoring` (`str`): Scoring metric used by `GridSearchCV`.

## Main Attributes

- `pipeline`: Full training pipeline.
- `grid_search`: Fitted `GridSearchCV` object.
- `best_model`: Best trained model pipeline.
- `best_params`: Best hyperparameters found during grid search.

---

## Public Methods

### `fit(df_train: pd.DataFrame, param_grid: Optional[Dict[str, List[Any]]] = None) -> GridSearchCV`

Train the anomaly-aware regression model using `GridSearchCV`.

This method follows the same workflow as the baseline class:

1. validate the input DataFrame
2. separate features and target
3. build the preprocessing and modeling pipeline
4. create KFold cross-validation
5. run the hyperparameter search
6. save the best model and parameters

#### Parameters

- `df_train` (`pd.DataFrame`): Training DataFrame with features and target.
- `param_grid` (`Optional[Dict[str, List[Any]]]`): Custom hyperparameter grid. If not provided, the class uses its default grid.

#### Returns

- `GridSearchCV`: Fitted grid search object.

---

### `fit_with_best_params(df_train: pd.DataFrame, params_path: Optional[str] = None) -> Pipeline`

Train the model using already selected best hyperparameters.

This avoids running `GridSearchCV` again.

#### Parameters

- `df_train` (`pd.DataFrame`): Training DataFrame with features and target.
- `params_path` (`Optional[str]`): Path to a JSON file with previously saved best parameters.

#### Returns

- `Pipeline`: Trained pipeline with the loaded best parameters.

#### Important note

This method expects `self.best_params` to be available, either from a previous `fit()` call or from a loading step.

---

### `predict(df: pd.DataFrame) -> np.ndarray`

Generate predictions with the trained anomaly model.

#### Parameters

- `df` (`pd.DataFrame`): DataFrame with the required feature columns.

#### Returns

- `np.ndarray`: Predicted target values.

---

### `evaluate(df_test: pd.DataFrame) -> Dict[str, float]`

Evaluate the trained model on a test DataFrame.

The method returns:

- MAE
- RMSE
- R2

#### Parameters

- `df_test` (`pd.DataFrame`): Test DataFrame with features and target.

#### Returns

- `Dict[str, float]`: Evaluation metrics.

---

### `save_model(filename: str = "lightgbm_anomalies_model.joblib") -> Path`

Save the trained anomaly model to disk.

#### Parameters

- `filename` (`str`): Output model file name.

#### Returns

- `Path`: Full path of the saved model.

---

### `save_best_params(filename: str = "lightgbm_anomalies_best_params.json") -> Path`

Save the best hyperparameters and metadata to a JSON file.

The saved file includes:

- best parameters
- best score
- scoring metric
- number of folds
- target column
- numeric features
- categorical features

#### Parameters

- `filename` (`str`): Output JSON file name.

#### Returns

- `Path`: Full path of the saved JSON file.

---

### `save_metrics(metrics: Dict[str, float], filename: str = "lightgbm_anomalies_metrics.json") -> Path`

Save evaluation metrics to a JSON file.

#### Parameters

- `metrics` (`Dict[str, float]`): Dictionary with evaluation results.
- `filename` (`str`): Output metrics file name.

#### Returns

- `Path`: Full path of the saved metrics file.

---

### `load_model(model_path: str) -> None`

Load a previously trained anomaly model from disk.

#### Parameters

- `model_path` (`str`): Path to the saved joblib file.

#### Returns

- `None`

---

# Shared Pipeline Structure

Both classes use the same machine learning structure.

## Preprocessing

The pipeline preprocesses:

### Numeric features
- missing values are filled with the median

### Categorical features
- missing values are filled with the most frequent value
- values are encoded with `OneHotEncoder`

## Model

Both classes use:

```python
LGBMRegressor(
    objective="regression",
    random_state=self.random_state,
    n_jobs=-1,
    verbosity=-1,
    force_col_wise=True
)
```

## Cross-validation

Both classes use:

- `KFold`
- shuffled folds
- reproducible random state

## Hyperparameter tuning

Both classes use a default hyperparameter grid with parameters such as:

- `n_estimators`
- `learning_rate`
- `num_leaves`
- `max_depth`
- `subsample`
- `colsample_bytree`
- `min_child_samples`

---

# Expected Input Data

Both classes expect the input DataFrame to contain:

- all numeric feature columns
- all categorical feature columns
- the target column

If required columns are missing, the internal validation method raises a `ValueError`. fileciteturn5file0 fileciteturn5file1

---

# Example Usage

## Baseline model

```python
from LightGBMRegressorBaseLine import LightGBMRegressorBaseLine

baseline_model = LightGBMRegressorBaseLine(
    target_col="quantity_sum_next_7d",
    numeric_features=["quantity_sum", "avg_ticket"],
    categorical_features=["Product", "Region"]
)

baseline_model.fit(df_train)
metrics = baseline_model.evaluate(df_test)

print(metrics)

baseline_model.save_model()
baseline_model.save_best_params()
baseline_model.save_metrics(metrics)
```

## Anomaly-aware model

```python
from LightGBMRegressorAnomaly import LightGBMRegressorAnomaly

anomaly_model = LightGBMRegressorAnomaly(
    target_col="quantity_sum_next_7d",
    numeric_features=["quantity_sum", "avg_ticket", "anomaly_score"],
    categorical_features=["Product", "Region"]
)

anomaly_model.fit(df_train)
metrics = anomaly_model.evaluate(df_test)

print(metrics)

anomaly_model.save_model()
anomaly_model.save_best_params()
anomaly_model.save_metrics(metrics)
```

---

# Typical Project Workflow

A common project flow is:

1. clean the raw data
2. create business and temporal features
3. optionally generate anomaly features
4. train the baseline regressor
5. train the anomaly-aware regressor
6. compare metrics
7. save the selected model and metadata

---

# Important Notes

- `LightGBMRegressorBaseLine` is the baseline comparison model. fileciteturn5file1
- `LightGBMRegressorAnomaly` is intended for a richer feature space with anomaly-related signals. fileciteturn5file0
- Both classes use structured preprocessing and are suitable for tabular machine learning tasks.
- Both classes are designed for reproducible training and export.
- The `evaluate()` method returns MAE, RMSE, and R2. fileciteturn5file0 fileciteturn5file1

---
# Summary

This document describes two LightGBM regression classes used in the project.

- `LightGBMRegressorBaseLine` supports the baseline regression approach.
- `LightGBMRegressorAnomaly` supports a regression approach that includes anomaly-related information.

Together, they help compare modeling strategies in a clear and reproducible way.
