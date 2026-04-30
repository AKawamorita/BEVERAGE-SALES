# DataQualityFeatures

## Overview

`DataQualityFeatures` is a utility class used to build and validate time-based sales features for anomaly detection and temporal analysis.

This class helps transform raw transactional sales data into daily aggregated metrics, calendar features, rolling-window features, deviation features, future quantity targets, and quality checks.

It was designed to keep feature generation, validation, and correction separated, which makes the pipeline easier to understand and maintain.

---

## Main Purpose

The class supports the following tasks:

- Aggregate transactional sales data into daily metrics by business group
- Create rolling features using only past data
- Avoid data leakage with `shift(1)` in historical windows
- Detect missing derived feature values
- Validate calendar and aggregated business metrics
- Rebuild or fix generated features when needed
- Provide a compact executive summary of feature quality

---

## Expected Input Columns

The input DataFrame should contain these columns:

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

---

## Constructor

### `DataQualityFeatures(df, date_col="Order_Date", group_cols=None, fill_missing_days=True)`

Create a new instance of the class.

**Parameters**

- `df` (`pd.DataFrame`): Input transactional DataFrame.
- `date_col` (`str`, optional): Name of the date column. Default is `"Order_Date"`.
- `group_cols` (`list | None`, optional): Columns used to group the data. If `None`, the default is `[`Product`, `Region`]`.
- `fill_missing_days` (`bool`, optional): If `True`, missing calendar days inside each group are added before creating rolling features. Default is `True`.

**Attributes**

- `original_df`: Original copy of the input DataFrame.
- `df`: Working copy of the input DataFrame.
- `features_df`: DataFrame with generated features.
- `report`: DataFrame with quality analysis results.
- `date_col`: Name of the date column.
- `group_cols`: Grouping columns.
- `fill_missing_days`: Flag used in the full pipeline.

---

## Generated Features

### Base Daily Features

- `quantity_sum`
- `total_price_sum`
- `unit_price_mean`
- `discount_mean`
- `order_count`
- `customer_count`
- `avg_ticket`

### Calendar Features

- `day_of_week`
- `month`
- `year`
- `is_weekend`

### Rolling Features

- `quantity_sum_mean_7d`
- `quantity_sum_std_7d`
- `quantity_sum_sum_7d`
- `total_price_sum_mean_7d`
- `total_price_sum_std_7d`
- `total_price_sum_mean_30d`
- `unit_price_mean_mean_7d`
- `discount_mean_mean_14d`

### Deviation Features

- `quantity_vs_mean_7d`
- `total_price_vs_mean_30d`
- `quantity_pct_vs_mean_7d`

### Future Quantity Targets

- `quantity_sum_next_7d`
- `quantity_sum_next_14d`
- `quantity_sum_next_30d`

### History Flags

- `history_less_than_7d`
- `history_less_than_30d`

---

## Public Methods

### `build_base_daily_features()`

Aggregate the raw transactional data into daily business metrics.

**Returns**

- `pd.DataFrame`: DataFrame with daily aggregated features.

**What it does**

- Validates required columns
- Parses the date column
- Converts numeric columns safely
- Groups data by business keys and date
- Creates daily metrics such as quantity, revenue, and ticket average

---

### `fill_missing_days_by_group()`

Fill missing calendar days inside each group.

**Returns**

- `pd.DataFrame`: DataFrame with continuous daily rows for each group.

**What it does**

- Builds a complete daily calendar between the minimum and maximum date of each group
- Reindexes missing dates
- Fills base metric columns with zero

**Why it is useful**

This step helps rolling windows work on continuous daily timelines.

---

### `create_calendar_features()`

Create simple calendar-based features.

**Returns**

- `pd.DataFrame`: DataFrame with calendar columns.

**Generated columns**

- `day_of_week`
- `month`
- `year`
- `is_weekend`

---

### `create_rolling_features()`

Create rolling-window features using only past information.

**Returns**

- `pd.DataFrame`: DataFrame with rolling and deviation features.

**What it does**

- Uses grouped rolling windows
- Applies `shift(1)` before rolling calculations
- Creates mean, standard deviation, and sum features
- Creates deviation and percentage deviation features

**Important**

The use of `shift(1)` avoids data leakage because the current row does not use its own value inside the historical window.

---

### `create_future_quantity_targets()`

Create future quantity targets for supervised modeling.

**Returns**

- `pd.DataFrame`: DataFrame with future quantity target columns.

**Generated columns**

- `quantity_sum_next_7d`
- `quantity_sum_next_14d`
- `quantity_sum_next_30d`

**Important**

These columns look into the future and should not be used as input features for anomaly detection.

---

### `create_history_flags()`

Create flags that indicate insufficient historical depth.

**Returns**

- `pd.DataFrame`: DataFrame with history flag columns.

**Generated columns**

- `history_less_than_7d`
- `history_less_than_30d`

**Why it is useful**

These flags help identify rows created with limited historical context.

---

### `fill_initial_feature_nans()`

Fill missing values in derived temporal features caused by limited initial history.

**Returns**

- `pd.DataFrame`: DataFrame with corrected missing values.

**Strategy**

- Rolling means are filled with `0`
- Rolling sums are filled with `0`
- Rolling standard deviations are filled with `0`
- Deviation values are filled with `0`
- Percentage deviation values are filled with `0`

---

### `build_all_features()`

Run the full feature engineering pipeline.

**Returns**

- `pd.DataFrame`: Final feature DataFrame.

**Pipeline steps**

1. Build base daily features
2. Fill missing days by group, if enabled
3. Create calendar features
4. Create rolling features
5. Create history flags
6. Create future quantity targets
7. Fill initial missing values in derived features

---

## Feature Quality Analysis Methods

### `analyze_missing_feature_values()`

Check whether derived feature columns still contain missing values.

**Returns**

- `dict`: Report row with issue name, count, severity, and recommendation.

---

### `analyze_negative_base_metrics()`

Check whether aggregated business metrics contain negative values.

**Returns**

- `dict`: Report row with issue name, count, severity, and recommendation.

**Checked columns**

- `quantity_sum`
- `total_price_sum`
- `order_count`
- `customer_count`
- `avg_ticket`

---

### `analyze_invalid_calendar_features()`

Validate the expected ranges of calendar features.

**Returns**

- `dict`: Report row with issue name, count, severity, and recommendation.

**Validation rules**

- `day_of_week` must be between `0` and `6`
- `month` must be between `1` and `12`
- `is_weekend` must be `0` or `1`

---

### `run_full_analysis()`

Run the complete feature quality analysis.

**Returns**

- `pd.DataFrame`: Compact quality report.

**Included checks**

- Missing derived feature values
- Negative aggregated metrics
- Invalid calendar features

---

## Fix Methods

### `fix_missing_feature_values()`

Reapply missing-value correction to derived temporal features.

**Returns**

- `pd.DataFrame`: Corrected feature DataFrame.

---

### `rebuild_calendar_features()`

Recompute calendar features.

**Returns**

- `pd.DataFrame`: Feature DataFrame with rebuilt calendar columns.

---

### `rebuild_rolling_features()`

Recompute rolling features and apply missing-value correction again.

**Returns**

- `pd.DataFrame`: Updated feature DataFrame.

---

### `rebuild_history_flags()`

Recompute historical depth flags.

**Returns**

- `pd.DataFrame`: Updated feature DataFrame.

---

## Reporting and Access Methods

### `generate_executive_summary()`

Return a compact text summary of the quality checks.

**Returns**

- `str`: Human-readable executive summary.

**Summary includes**

- Number of evaluated rules
- Number of rules with detected issues
- Total issue count
- High-severity issue names

---

### `get_feature_dataframe()`

Return the engineered feature DataFrame.

**Returns**

- `pd.DataFrame`: Copy of the generated feature DataFrame.

**Behavior**

If features were not created yet, the method runs the full feature pipeline automatically.

---

### `reset_to_original()`

Reset the object to its original state.

**Returns**

- `None`

**What it resets**

- Working DataFrame
- Generated feature DataFrame
- Analysis report

---

## Example Usage

```python
from data_quality_features import DataQualityFeatures

feature_builder = DataQualityFeatures(df)

# Build all features
features = feature_builder.build_all_features()

# Run quality analysis
report = feature_builder.run_full_analysis()

# Generate text summary
summary = feature_builder.generate_executive_summary()

# Fix missing derived values if needed
features_fixed = feature_builder.fix_missing_feature_values()
```

---


