# Beverage Sales Project — Notebook Documentation

![main cities and products consumed](img/InfoGraphic50.png)

## Overview

This document describes the main notebooks used in the **Beverage Sales** project. The pipeline follows a practical machine learning flow for tabular and time-based sales data:

1. Load raw CSV data
2. Save the dataset in Parquet format
3. Validate data quality
4. Create time-based features
5. Detect anomalies with Isolation Forest
6. Train a baseline model with LightGBM

The notebooks were designed to run in sequence, sharing intermediate artifacts through Parquet files and serialized models.

- sequential organization of the data pipeline
- explicit data quality step
- time-based feature engineering
- careful time split to avoid leakage
- combination of supervised and unsupervised modeling ideas

---

## Recommended Execution Order

1. `data_loading.ipynb`
2. `data_quality.ipynb`
3. `feature_engineering.ipynb`
4. `anomaly_detection.ipynb`
5. `BaseLine.ipynb`

---

## 1) `data_loading.ipynb`

### Purpose
This notebook loads the raw CSV dataset and saves it in Parquet format, allowing faster and more consistent processing in the next steps.

### Main Components
- `CSVLoader`
- `ParquetRepository`
- `DATA_RAW`
- `DATA_PROCESSED`

### What the Notebook Does
- Imports project paths and helper classes
- Loads the file `synthetic_beverage_sales_data.csv`
- Creates a pandas DataFrame from the raw source
- Saves the result as `beverage_sales_processed.parquet`
- Reloads the saved Parquet file
- Checks whether the loaded file is equal to the original DataFrame

### Input
- Raw CSV file from the project's raw data folder

### Output
- `beverage_sales_processed.parquet`

### Why This Step Is Important
- Reduces loading time in the next steps
- Standardizes the storage format
- Improves pipeline reproducibility

### Notes
- This notebook is the entry point of the data pipeline.
- The equality check is useful to validate the save and reload process.

---

## 2) `data_quality.ipynb`

### Purpose
This notebook validates the processed dataset before the feature engineering and modeling stage.

### Main Components
- `ParquetRepository`
- `DataQuality`
- `DATA_PROCESSED`

### What the Notebook Does
- Loads `beverage_sales_processed.parquet`
- Creates an instance of the `DataQuality` class
- Runs a full analysis with `run_full_analysis(tolerance=0.1)`
- Displays the validation report
- Shows a sample of total price inconsistencies
- Generates an executive summary
- Displays the DataFrame column types

### Input
- `beverage_sales_processed.parquet`

### Output
- Quality report shown in the notebook
- Executive summary shown in the notebook
- Sample of inconsistent records shown in the notebook

### Expected Checks in This Step
Depending on the implementation of the `DataQuality` class, this step may validate:
- Missing values
- Negative values
- Discount rules
- Duplicate records
- Invalid dates
- Inconsistencies between total price and expected total price

### Why This Step Is Important
- Prevents low-quality records from affecting feature engineering
- Gives more confidence to the modeling pipeline
- Makes business rules more explicit

### Notes
- The tolerance parameter is important for numeric comparisons.
- This notebook is a validation step and should be run before creating derived features.

---

## 3) `feature_engineering.ipynb`

### Purpose
This notebook creates time-based and aggregated features for the sales dataset. It also includes exploratory visual analysis of distributions and outliers.

### Main Components
- `ParquetRepository`
- `DataQualityFeatures`
- `SalesTimeBoxplot`
- `SalesTimeViolin`
- `SalesTimeOutlier`
- `ProductQuantityHistogram`
- `DATA_PROCESSED`
- `DATA_FEATURES`

### Input File
- `beverage_sales_processed_qa_ok.parquet`

### What the Notebook Does
- Loads the processed dataset approved in the quality validation step
- Creates a `DataQualityFeatures` object using:
  - `date_col="Order_Date"`
  - `group_cols=["Category", "Product", "Region"]`
  - `fill_missing_days=True`
- Generates all features with `build_all_features()`
- Checks missing values in important derived columns
- Displays data types and first rows
- Runs feature quality analysis and executive summary
- Compares the size of the original DataFrame with the enriched DataFrame
- Merges original data with feature-enriched data
- Checks key duplication after the merge
- Creates visual analysis with:
  - boxplot by period
  - violin plot by period
  - outlier chart using IQR
  - outlier chart using z-score
  - histogram by year/category/product
- Saves the final dataset with features as `beverage_sales_feature.parquet`

### Examples of Features Found in This Notebook
The generated dataset includes features such as:
- `quantity_sum_mean_7d`
- `quantity_sum_std_7d`
- `quantity_sum_sum_7d`
- `total_price_sum_mean_7d`
- `total_price_sum_std_7d`
- `total_price_sum_mean_30d`
- `unit_price_mean_mean_7d`
- `discount_mean_mean_14d`
- `quantity_vs_mean_7d`
- `total_price_vs_mean_30d`
- `quantity_pct_vs_mean_7d`
- `history_less_than_7d`
- `history_less_than_30d`

### Output
- `beverage_sales_feature.parquet`
- Feature quality report in the notebook
- Visual analysis charts for distribution and outliers

### Why This Step Is Important
- Adds time context to each sales record
- Helps models understand recent behavior and local trends
- Creates stronger signals for anomaly detection and forecasting

### Notes
- The grouping keys are important because rolling statistics are calculated inside each group.
- The notebook uses `fill_missing_days=True`, which is useful in time series because it makes the historical window more consistent.
- There are exploratory analysis cells, but the main production artifact of this step is the Parquet file with the features.

### Points of Attention
- Some visualization cells use `value_col="Quantity"`, while the feature dataset mainly uses aggregated names such as `quantity_sum`. This is acceptable only if the plotting class expects the original column. Otherwise, it is worth reviewing.
- One violin plot uses an end date in 2024. If the dataset is limited to the 2021 to 2023 period, this filter should be checked.

---

## 4) `anomaly_detection.ipynb`

### Purpose
This notebook trains an **Isolation Forest** model to detect anomalous sales behavior and generate anomaly signals for later use.

### Main Components
- `ParquetRepository`
- `IsolationForestAnalyzer`
- `DATA_FEATURES`
- `MODELS_ANOMALYD`

### Input
- `beverage_sales_feature.parquet`

### What the Notebook Does
- Loads the dataset with features
- Converts `Order_Date` to datetime
- Splits the data by year:
  - Training: 2021 and 2022
  - Test / future period: 2023
- Creates an `IsolationForestAnalyzer` with:
  - `random_state=42`
  - `cv=3`
  - `n_jobs=-1`
  - `verbose=1`
- Trains the model only on the training period
- Displays the best parameters
- Saves the trained model to a `.joblib` file
- Saves the best parameters to JSON
- Runs prediction on the feature dataset
- Selects anomaly-related columns, including:
  - `anomaly_flag`
  - `anomaly_label`
  - `anomaly_score`
  - `if_qty_signal`
  - `if_sales_signal`
  - `if_discount_signal`
  - `if_ticket_signal`
- Displays an anomaly summary

### Output
- Serialized model: `isolation_forest_analyzer.joblib`
- Parameter file: `isolation_forest_best_params.json`
- Anomaly predictions available in memory / notebook output

### Why This Step Is Important
- Detects unusual patterns without needing labels
- Creates useful anomaly signals that can be used later by supervised models
- Supports a hybrid modeling strategy

### Data Leakage Prevention
This notebook clearly defines an important rule:

> To avoid data leakage, the Isolation Forest model is trained only on data from 2021 to 2022, leaving 2023 for testing.

This is a strong and correct decision for time-based modeling.

### Notes
- Although `df_test_if` is created, prediction is made on `df_load`. This may be intentional if the goal is to score the full dataset after training only on historical data.
- The anomaly output columns are useful for future comparison between the baseline model and the anomaly-enriched model.

---

## 5) `BaseLine.ipynb`

### Purpose
This notebook trains a **baseline model with LightGBM Regressor** to predict `quantity_sum`, without using anomaly-derived signals yet.

### Main Components
- `ParquetRepository`
- `LightGBMRegressorBaseLine`
- `LightGBMRegressorAnomaly` (imported, but not used in the visible cells)
- `DATA_FEATURES`
- `MODELS_BASELINE`
- `MODELS_METRICS`

### Input
- `beverage_sales_feature.parquet`

### What the Notebook Does
- Loads the dataset with features
- Displays the column types
- Converts `Order_Date` to datetime
- Splits the data by year:
  - Training: 2021 and 2022
  - Test: 2023
- Defines the target:
  - `quantity_sum`
- Defines numeric features, such as:
  - price, discount, order and customer aggregations
  - calendar features
  - rolling window features
  - ratio and history features
- Defines categorical features:
  - `Product`
  - `Region`
- Creates an instance of `LightGBMRegressorBaseLine`
- Trains the model with cross-validation and hyperparameter search
- Evaluates the model
- Saves the trained model and best parameters

### Output
- Serialized baseline model
- File with best parameters
- Evaluation metrics shown in the notebook

### Why This Step Is Important
- Establishes a supervised benchmark
- Creates a reference point before using anomaly-derived signals
- Helps compare a simple forecasting approach with a hybrid approach

### Notes
- In the visible section, the evaluation is done with `baseline_model.evaluate(df_train_baseline)`. For the final performance report, the most appropriate choice is to evaluate on `df_test_baseline`.
- This notebook is important because it provides the baseline needed to justify whether the anomaly-based approach really adds value.

---

## Pipeline Summary

### End-to-End Logic
The notebooks implement the following practical flow:

- **Load and standardize the data**
- **Validate business consistency in the data**
- **Create time-based and rolling features**
- **Detect unusual patterns with an unsupervised model**
- **Train a supervised baseline forecasting model**

### Main Generated Artifacts
- `beverage_sales_processed.parquet`
- `beverage_sales_feature.parquet`
- Data quality report
- Feature quality report
- Serialized Isolation Forest model
- JSON with best Isolation Forest parameters
- Serialized LightGBM baseline model
- File with best LightGBM baseline parameters

---

## Note

These notebooks already show a coherent ML pipeline structure for a portfolio project. The strongest points are:
- sequential organization of the data pipeline
- explicit data quality step
- time-based feature engineering
- careful time split to avoid leakage
- combination of supervised and unsupervised modeling ideas
