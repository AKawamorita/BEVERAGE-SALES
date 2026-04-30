# IsolationForestAnalyzer

## Overview

`IsolationForestAnalyzer` is a class used to detect anomalies in aggregated sales data with the Isolation Forest algorithm.

The class prepares anomaly signals from business and temporal features, trains an unsupervised model, predicts anomalies, creates summaries, shows plots, and exports reports.

It is useful in machine learning projects that need to identify unusual sales behavior, price changes, discount patterns, or abnormal ticket values.

---

## Main Purpose

This class helps build an anomaly detection workflow with the following steps:

- create model-ready anomaly signals
- train an Isolation Forest model with hyperparameter search
- predict anomaly flags and anomaly scores
- inspect the most severe anomalies
- visualize anomalies over time
- summarize anomalies by business group
- export reports
- save and load trained models

---

## Class: `IsolationForestAnalyzer`

### Constructor

```python
IsolationForestAnalyzer(
    base_path: str = None,
    random_state=42,
    cv=3,
    n_jobs=-1,
    verbose=0
)
```

### Parameters

- `base_path` (`str`, optional): Base folder used to save model files and parameter files.
- `random_state` (`int`, optional): Random seed used by the model. Default is `42`.
- `cv` (`int`, optional): Number of cross-validation folds used in grid search. Default is `3`.
- `n_jobs` (`int`, optional): Number of parallel jobs used in grid search. Default is `-1`.
- `verbose` (`int`, optional): Verbosity level used in grid search. Default is `0`.

### Main Attributes

- `feature_cols_`: Model input feature names used by the analyzer.
- `pipeline_`: Scikit-learn pipeline created during training.
- `grid_search_`: Grid search object used during hyperparameter tuning.
- `best_estimator_`: Best fitted model pipeline.
- `best_params_`: Best hyperparameters found during training.
- `fitted_`: Boolean flag that indicates whether the model has already been trained.

---

## Model Input Signals

The class creates and uses the following anomaly signals:

- `if_qty_signal`
- `if_sales_signal`
- `if_discount_signal`
- `if_ticket_signal`

These signals are derived from historical sales behavior and business metrics.

### Source columns used to create signals

- `quantity_pct_vs_mean_7d`
- `total_price_vs_mean_30d`
- `discount_mean_mean_14d`
- `avg_ticket`

---

## Public Methods

## Configuration and Utility Methods

### `set_base_path(base_path: str = "data")`

Redefine the base folder used to store saved model files.

### Parameters

- `base_path` (`str`, optional): New base directory.

**Returns:**
- `None`

---

### `filter_dataframe(df: pd.DataFrame, product_filter=None, region_filter=None) -> pd.DataFrame`

Filter a DataFrame by `Product` and/or `Region`.

This method accepts:

- `None`
- a single value
- a list of values

### Parameters

- `df` (`pd.DataFrame`): Input DataFrame.
- `product_filter` (optional): One product or a list of products.
- `region_filter` (optional): One region or a list of regions.

**Returns:**
- `pd.DataFrame`: Filtered DataFrame.

---

## Feature Creation Method

### `create_if_features(df: pd.DataFrame) -> pd.DataFrame`

Create the anomaly input signals used by the Isolation Forest model.

This method converts source columns to numeric values, handles infinite values, and clips extreme values to reduce instability.

### Required input columns

- `quantity_pct_vs_mean_7d`
- `total_price_vs_mean_30d`
- `discount_mean_mean_14d`
- `avg_ticket`

### Output columns created

- `if_qty_signal`
- `if_sales_signal`
- `if_discount_signal`
- `if_ticket_signal`

**Returns:**
- `pd.DataFrame`: DataFrame with the new anomaly signal columns.

---

## Training and Prediction Methods

### `fit(df: pd.DataFrame)`

Train the analyzer with grid search and unsupervised scoring.

This method performs the following steps:

1. create Isolation Forest input signals
2. build a preprocessing and modeling pipeline
3. run `GridSearchCV`
4. store the best fitted estimator
5. store the best hyperparameters

The internal pipeline includes:

- `SimpleImputer(strategy="median")`
- `RobustScaler()`
- `IsolationForest(...)`

**Returns:**
- `IsolationForestAnalyzer`: The fitted object itself.

---

### `predict(df: pd.DataFrame) -> pd.DataFrame`

Predict anomalies for a DataFrame using the trained model.

The method adds the following columns to the output:

- `anomaly_flag`
- `anomaly_label`
- `anomaly_score`

### Output columns

- `anomaly_flag`: `1` for anomaly and `0` for normal observation.
- `anomaly_label`: `"Anomaly"` or `"Normal"`.
- `anomaly_score`: Decision function score from the fitted model.

**Returns:**
- `pd.DataFrame`: Input data plus anomaly prediction columns.

---

### `get_best_params() -> dict`

Return the best hyperparameters found during training.

**Returns:**
- `dict`: Best parameter dictionary.

---

## Analysis and Inspection Methods

### `top_n_anomalies(df_pred: pd.DataFrame, n: int = 10, product_filter=None, region_filter=None, columns_to_show: list = None) -> pd.DataFrame`

Return the `n` most severe anomalies.

The method filters the predicted dataset, keeps only anomaly rows, and sorts them by anomaly score in ascending order.

Lower anomaly scores indicate stronger anomaly behavior.

### Parameters

- `df_pred` (`pd.DataFrame`): DataFrame already processed by `predict`.
- `n` (`int`, optional): Number of anomaly rows to return. Default is `10`.
- `product_filter` (optional): Optional filter by product.
- `region_filter` (optional): Optional filter by region.
- `columns_to_show` (`list`, optional): Custom list of columns to return.

**Returns:**
- `pd.DataFrame`: Top anomaly rows.

---

### `anomaly_summary(df_pred: pd.DataFrame, product_filter=None, region_filter=None) -> pd.DataFrame`

Create a grouped summary of anomalies by `Product` and `Region`.

The summary includes:

- `total_rows`
- `anomaly_count`
- `anomaly_rate`

### Parameters

- `df_pred` (`pd.DataFrame`): DataFrame already processed by `predict`.
- `product_filter` (optional): Optional filter by product.
- `region_filter` (optional): Optional filter by region.

**Returns:**
- `pd.DataFrame`: Summary table sorted by anomaly count and anomaly rate.

---

## Visualization Methods

### `plot_anomalies_over_time(df_pred: pd.DataFrame, date_col: str = "Order_Date", value_col: str = "total_price_sum", title: str = "Anomalies Over Time", figsize: tuple = (14, 6), product_filter=None, region_filter=None)`

Plot a time series and highlight anomaly points.

Normal observations are shown as a line.  
Anomalies are shown as highlighted scatter points.

### Parameters

- `df_pred` (`pd.DataFrame`): DataFrame already processed by `predict`.
- `date_col` (`str`, optional): Date column used on the x-axis. Default is `"Order_Date"`.
- `value_col` (`str`, optional): Metric plotted on the y-axis. Default is `"total_price_sum"`.
- `title` (`str`, optional): Plot title.
- `figsize` (`tuple`, optional): Figure size.
- `product_filter` (optional): Optional filter by product.
- `region_filter` (optional): Optional filter by region.

**Returns:**
- `None`

---

### `plot_anomalies_subplots(df_pred: pd.DataFrame, group_by: str = "Product", date_col: str = "Order_Date", value_col: str = "total_price_sum", product_filter=None, region_filter=None, max_groups: int = 6, figsize_per_plot: tuple = (14, 4))`

Plot anomaly time series in separate subplots by `Product` or `Region`.

This method is useful when the user wants to compare anomaly behavior across different groups.

### Parameters

- `df_pred` (`pd.DataFrame`): DataFrame already processed by `predict`.
- `group_by` (`str`, optional): Subplot grouping column. Accepted values are `"Product"` or `"Region"`.
- `date_col` (`str`, optional): Date column used on the x-axis.
- `value_col` (`str`, optional): Metric plotted on the y-axis.
- `product_filter` (optional): Optional filter by product.
- `region_filter` (optional): Optional filter by region.
- `max_groups` (`int`, optional): Maximum number of groups shown in the figure. Default is `6`.
- `figsize_per_plot` (`tuple`, optional): Base size used for each subplot.

**Returns:**
- `None`

---

## Export Methods

### `export_anomaly_report(df_pred: pd.DataFrame, folder_path: str = "reports", file_name: str = None, top_n: int = 20, product_filter=None, region_filter=None) -> str`

Export a CSV report containing the top anomalies.

If `file_name` is not provided, the method creates a timestamped file name automatically.

### Parameters

- `df_pred` (`pd.DataFrame`): DataFrame already processed by `predict`.
- `folder_path` (`str`, optional): Output folder. Default is `"reports"`.
- `file_name` (`str`, optional): Custom CSV file name.
- `top_n` (`int`, optional): Number of top anomalies exported. Default is `20`.
- `product_filter` (optional): Optional filter by product.
- `region_filter` (optional): Optional filter by region.

**Returns:**
- `str`: Full path of the exported CSV file.

---

## Save and Load Methods

### `save_model(folder_path=None, file_name=None) -> str`

Save the full trained analyzer object as a `.joblib` file.

If no file name is provided, the method creates a timestamped file name automatically.

### Parameters

- `folder_path` (optional): Output folder.
- `file_name` (optional): Custom file name.

**Returns:**
- `str`: Full saved file path.

---

### `save_best_params_json(folder_path=None, file_name="isolation_forest_best_params.json") -> str`

Save the best hyperparameters found during training to a JSON file.

### Parameters

- `folder_path` (optional): Output folder.
- `file_name` (`str`, optional): JSON file name.

**Returns:**
- `str`: Full saved file path.

---

### `load_model(model_path: str)`

Load a previously saved analyzer object from disk.

This method is static and can be called without creating a new class instance first.

### Parameters

- `model_path` (`str`): Path to the saved `.joblib` file.

**Returns:**
- Loaded analyzer object.

---

## Example Usage

```python
from IsolationForest import IsolationForestAnalyzer

analyzer = IsolationForestAnalyzer()

analyzer.fit(df_features)

print(analyzer.get_best_params())

df_pred = analyzer.predict(df_features)

top_anomalies = analyzer.top_n_anomalies(df_pred, n=10)
print(top_anomalies)

summary = analyzer.anomaly_summary(df_pred)
print(summary)

analyzer.plot_anomalies_over_time(df_pred)

model_path = analyzer.save_model()
params_path = analyzer.save_best_params_json()
```

---

## Typical Workflow

A common workflow with this class is:

1. load and clean transactional data
2. build temporal features
3. train `IsolationForestAnalyzer`
4. predict anomalies
5. inspect top anomalies
6. visualize results
7. export reports
8. save the trained model

---

## Notes

- This class is designed for aggregated sales data.
- It depends on previously created temporal and business features.
- It uses an unsupervised model, so no target label is required.
- It is useful for exploratory anomaly detection and business monitoring.
- The anomaly score comes from the Isolation Forest decision function.

---

## Important Considerations

### Required feature engineering first

Before using this class, the input DataFrame should already contain the required temporal features used to build the anomaly signals.

### Unsupervised tuning strategy

The model uses a custom unsupervised scorer inside `GridSearchCV`.  
This helps choose parameters based on separation between more extreme points and more normal points.

### Filtering support

Several public methods allow optional filtering by product and region.  
This is useful for focused business analysis.

### Model persistence

The class supports model saving and loading, which is useful when training is expensive and the same fitted model needs to be reused later.

