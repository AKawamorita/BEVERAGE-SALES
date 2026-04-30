# DataFeatures

## Overview

`DataFeatures` is a public function used to create daily sales features for time series analysis, anomaly detection, and machine learning models.

It receives a transactional sales DataFrame and returns a new DataFrame with:

- daily aggregated business metrics
- calendar features
- rolling window statistics
- comparison features versus recent history
- percentage variation features

This function is useful in projects that need temporal features built from raw sales transactions.

---

## Main Purpose

The function helps transform raw sales records into a structured daily feature table.

Typical use cases include:

- anomaly detection
- demand analysis
- forecasting support
- supervised machine learning
- time-based business analysis

It is especially useful when the dataset contains many transactions and the model needs historical context.

---

## Function: `DataFeatures`

```python
DataFeatures(
    df: pd.DataFrame,
    date_col: str = "Order_Date",
    group_cols: list = None,
    windows: tuple = (7, 14, 30),
    min_periods: int = 1,
    fill_missing_days: bool = True
) -> pd.DataFrame
```

---

## Parameters

### `df`
- Type: `pd.DataFrame`
- Description: Raw transactional sales DataFrame.

### `date_col`
- Type: `str`
- Default: `"Order_Date"`
- Description: Name of the column used as the transaction date.

### `group_cols`
- Type: `list`
- Default: `["Product", "Region"]`
- Description: Columns used to define the business grouping for daily aggregation and rolling windows.

### `windows`
- Type: `tuple`
- Default: `(7, 14, 30)`
- Description: Rolling window sizes, in days, used to create historical statistics.

### `min_periods`
- Type: `int`
- Default: `1`
- Description: Minimum number of past observations required to compute rolling features.

### `fill_missing_days`
- Type: `bool`
- Default: `True`
- Description: If `True`, the function creates missing calendar days inside each group so the daily time series becomes continuous.

---

## Returns

- `pd.DataFrame`: A daily feature DataFrame with aggregated, calendar, rolling, and comparison features.

---

## Expected Input Columns

The function requires the following columns:

- `Order_ID`
- `Customer_ID`
- `Customer_Type`
- `Product`
- `Category`
- `Unit_Price`
- `Quantity`
- `Discount`
- `Total_Price`
- `Region`
- `Order_Date`  
  or the column defined by `date_col`

If one or more required columns are missing, the function raises a `ValueError`.

---

## Processing Steps

## 1. Date conversion

The function converts the date column to datetime using safe parsing.

Invalid dates are converted to missing values and then removed.

---

## 2. Daily normalization

The datetime column is normalized to daily level, so all transactions from the same day share the same date value.

---

## 3. Daily aggregation

The function groups the raw data by:

- business grouping columns
- daily date

Then it creates daily metrics such as:

- `quantity_sum`
- `quantity_mean`
- `quantity_std`
- `total_price_sum`
- `total_price_mean`
- `unit_price_mean`
- `discount_mean`
- `order_count`
- `customer_count`

It also creates:

- `avg_ticket`

This feature is calculated as:

```python
total_price_sum / order_count
```

when `order_count` is greater than zero.

---

## 4. Missing day completion

When `fill_missing_days=True`, the function fills missing calendar days inside each business group.

For days without sales, it fills the main numeric metrics with zero.

This helps create continuous daily series, which is important for rolling windows.

---

## 5. Sorting

The resulting daily dataset is sorted by:

- grouping columns
- date column

This is necessary to ensure correct rolling calculations.

---

## 6. Calendar features

The function creates the following calendar features:

- `day_of_week`
- `day`
- `month`
- `year`
- `is_weekend`

These features help the model capture temporal and seasonal behavior.

---

## 7. Rolling features

Rolling features are built using only past data.

The function uses:

```python
shift(1)
```

before the rolling window calculation.

This is important because it avoids using the current day's value inside its own history window, which helps prevent data leakage.

Rolling statistics are created for these metrics:

- `quantity_sum`
- `total_price_sum`
- `unit_price_mean`
- `discount_mean`
- `order_count`
- `avg_ticket`

For each window size, the function creates:

- rolling mean
- rolling standard deviation
- rolling sum

Examples:

- `quantity_sum_mean_7d`
- `quantity_sum_std_7d`
- `quantity_sum_sum_7d`
- `total_price_sum_mean_14d`
- `avg_ticket_sum_30d`

---

## 8. Comparison with recent history

For each window size, the function creates deviation features that compare the current day with recent rolling averages.

Examples:

- `quantity_vs_mean_7d`
- `total_price_vs_mean_14d`
- `unit_price_vs_mean_30d`
- `discount_vs_mean_7d`

These features help show whether the current day is above or below recent historical behavior.

---

## 9. Percentage variation features

The function also creates percentage-based comparison features for quantity.

Example:

- `quantity_pct_vs_mean_7d`

This value shows how much the current quantity is above or below the recent rolling mean, in percentage form.

When the rolling mean is zero, the result is set to `NaN`.

---

## 10. Standard deviation fill

Some rolling standard deviation columns may be missing at the beginning of the series.

The function fills these missing rolling standard deviation values with zero.

---

## Main Output Features

Below is a summary of the main feature groups created by the function.

## Daily business metrics

- `quantity_sum`
- `quantity_mean`
- `quantity_std`
- `total_price_sum`
- `total_price_mean`
- `unit_price_mean`
- `discount_mean`
- `order_count`
- `customer_count`
- `avg_ticket`

## Calendar features

- `day_of_week`
- `day`
- `month`
- `year`
- `is_weekend`

## Rolling features

For each window in `windows`, the function creates rolling features such as:

- `{metric}_mean_{w}d`
- `{metric}_std_{w}d`
- `{metric}_sum_{w}d`

where `metric` may be:

- `quantity_sum`
- `total_price_sum`
- `unit_price_mean`
- `discount_mean`
- `order_count`
- `avg_ticket`

## Comparison features

For each window in `windows`, the function creates:

- `quantity_vs_mean_{w}d`
- `total_price_vs_mean_{w}d`
- `unit_price_vs_mean_{w}d`
- `discount_vs_mean_{w}d`

## Percentage variation features

For each window in `windows`, the function creates:

- `quantity_pct_vs_mean_{w}d`

---

## Example Usage

```python
from data_features import DataFeatures

df_features = DataFeatures(
    df=df_sales,
    date_col="Order_Date",
    group_cols=["Product", "Region"],
    windows=(7, 14, 30),
    min_periods=1,
    fill_missing_days=True
)

print(df_features.head())
```

---

## Example in a Pipeline

A common pipeline can be:

1. load raw transactional data
2. apply data quality checks
3. call `DataFeatures(...)`
4. send the output to anomaly detection or a supervised model
5. save the final feature table

---

## Notes

- The function is designed for transactional sales datasets.
- It creates daily features at group level.
- It uses only past information for rolling windows.
- It helps reduce leakage risk in machine learning pipelines.
- Missing days can be filled automatically to improve temporal continuity.

---

## Important Considerations

### Data leakage prevention

The function uses `shift(1)` before rolling calculations.  
This means the rolling window only uses previous days, not the current day.

This is a good practice for machine learning and time series feature engineering.

### Zero-filled days

When `fill_missing_days=True`, days without transactions are added and numeric sales metrics are filled with zero.

This is useful in many business scenarios, but it should still be reviewed according to the project's business meaning.

### Initial rolling values

At the start of each series, rolling features may be based on very short history because `min_periods=1` by default.

This is practical, but users should still evaluate if this behavior fits the modeling strategy.


