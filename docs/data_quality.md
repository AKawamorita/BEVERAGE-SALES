# Data Quality Validation and Correction

## Overview

`DataQuality` is a Python class created to analyze and correct common data quality issues in beverage sales datasets.

The class separates two main responsibilities:

- **Analysis**: identify inconsistencies and generate a report
- **Correction**: apply controlled fixes to the dataset

It was designed to support data preparation workflows before feature engineering, anomaly detection, or machine learning modeling.

---

## Main Goals

This class helps with the following tasks:

- detect missing and invalid numeric values
- validate discount rules
- validate customer type values
- detect invalid dates
- detect duplicate rows
- validate `Total_Price` consistency
- apply corrections in a controlled way
- generate a simple executive summary of the results

---

## Class Definition

```python
DataQuality(df: pd.DataFrame)
```

### Parameters

#### `df`
A pandas DataFrame containing the beverage sales data to be analyzed.

### Behavior

When the class is initialized:

- `original_df` stores a copy of the original dataset
- `df` stores the working copy used for analysis and corrections
- `report` starts as `None` and is filled after running the full analysis

---

## Public Methods

## Analysis Methods

### `analyze_quantity_missing() -> dict`

Checks whether the `Quantity` column has missing values.

**Returns:**  
A dictionary representing one analysis rule result with:

- `rule_name`
- `issue_found`
- `issue_count`
- `severity`
- `recommendation`

**Rule meaning:**  
Missing quantities may indicate incomplete sales records.

---

### `analyze_quantity_negative() -> dict`

Checks whether the `Quantity` column has negative values.

**Returns:**  
A dictionary with the analysis result.

**Rule meaning:**  
Negative quantity is usually invalid in this type of sales dataset and should be reviewed.

---

### `analyze_discount_missing() -> dict`

Checks whether the `Discount` column has missing values.

**Returns:**  
A dictionary with the analysis result.

**Rule meaning:**  
Missing discount values may be treated as zero depending on business rules.

---

### `analyze_discount_negative() -> dict`

Checks whether the `Discount` column has negative values.

**Returns:**  
A dictionary with the analysis result.

**Rule meaning:**  
Negative discounts are usually not valid and should be corrected.

---

### `analyze_discount_above_one() -> dict`

Checks whether the `Discount` column has values above `1`.

**Returns:**  
A dictionary with the analysis result.

**Rule meaning:**  
Discount is expected to be in the range `[0, 1]`, where `1` means 100%.

---

### `analyze_customer_type_invalid() -> dict`

Checks whether `Customer_Type` contains values outside the expected set:

- `B2B`
- `B2C`

**Returns:**  
A dictionary with the analysis result.

**Rule meaning:**  
Customer type values should be standardized for business consistency.

---

### `analyze_b2c_discount_rule() -> dict`

Checks whether a `B2C` record has a discount greater than zero.

**Returns:**  
A dictionary with the analysis result.

**Rule meaning:**  
According to the implemented business rule, `B2C` records should not have discount values above zero.

---

### `analyze_order_date_invalid() -> dict`

Checks whether `Order_Date` contains invalid or non-parsable dates.

**Returns:**  
A dictionary with the analysis result.

**Rule meaning:**  
Date values must be valid to support time-based analysis, rolling windows, and forecasting.

---

### `analyze_duplicates() -> dict`

Checks for exact duplicate rows in the dataset.

**Returns:**  
A dictionary with the analysis result.

**Rule meaning:**  
Duplicate rows may distort metrics, reports, and model training.

---

### `analyze_total_price_consistency(tolerance: float = 0.1) -> dict`

Validates whether `Total_Price` is consistent with:

```python
Unit_Price * Quantity * (1 - Discount)
```

**Parameters:**

#### `tolerance`
Maximum allowed difference between expected and actual total price.  
Default is `0.1`.

**Returns:**  
A dictionary with the analysis result.

**Rule meaning:**  
This method checks whether the stored sales value matches the expected calculated value, allowing a small tolerance for rounding differences.

---

### `run_full_analysis(tolerance: float = 0.01) -> pd.DataFrame`

Runs all analysis methods and creates a full report.

**Parameters:**

#### `tolerance`
Tolerance used in the `Total_Price` consistency validation.  
Default is `0.01`.

**Returns:**  
A pandas DataFrame with the final analysis report.

**Notes:**

- one row is generated for each rule
- the `recommendation` column is removed from the final report
- the result is also stored in `self.report`

---

### `get_total_price_inconsistencies_sample(tolerance: float = 0.1, n: int = 20) -> pd.DataFrame`

Returns a sample of rows where `Total_Price` is inconsistent with the expected value.

**Parameters:**

#### `tolerance`
Maximum allowed difference between expected and actual total price.

#### `n`
Maximum number of rows to return.  
Default is `20`.

**Returns:**  
A DataFrame containing selected inconsistent rows, including:

- `Product`
- `Region`
- `Order_Date`
- numeric versions of the price fields
- `Expected_Total`
- `Diff`

**Use case:**  
Useful for manual inspection and debugging of pricing issues.

---

## Correction Methods

### `fix_quantity_missing_with_zero()`

Replaces missing values in `Quantity` with `0`.

**Returns:**  
The corrected working DataFrame.

**Use case:**  
Useful when missing quantity means no units sold.

---

### `fix_quantity_negative_with_zero()`

Replaces negative values in `Quantity` with `0`.

**Returns:**  
The corrected working DataFrame.

---

### `fix_discount_missing_with_zero()`

Replaces missing values in `Discount` with `0`.

**Returns:**  
The corrected working DataFrame.

---

### `fix_discount_negative_with_zero()`

Replaces negative values in `Discount` with `0`.

**Returns:**  
The corrected working DataFrame.

---

### `fix_discount_clip_range(lower: float = 0, upper: float = 1)`

Clips `Discount` values to a valid range.

**Parameters:**

#### `lower`
Lower allowed value. Default is `0`.

#### `upper`
Upper allowed value. Default is `1`.

**Returns:**  
The corrected working DataFrame.

---

### `fix_b2c_discount_rule()`

Sets `Discount` to `0` for rows where `Customer_Type == "B2C"`.

**Returns:**  
The corrected working DataFrame.

**Use case:**  
Applies the business rule that B2C sales must not have discount.

---

### `fix_order_date_parse()`

Converts `Order_Date` to datetime format using safe parsing.

**Returns:**  
The corrected working DataFrame.

**Use case:**  
Prepares the dataset for time series analysis and date filtering.

---

### `fix_remove_duplicates()`

Removes exact duplicate rows.

**Returns:**  
The corrected working DataFrame.

---

### `fix_total_price_recalculate(round_digits: int = 2)`

Recalculates `Total_Price` using:

```python
Unit_Price * Quantity * (1 - Discount)
```

**Parameters:**

#### `round_digits`
Number of decimal places used when rounding the recalculated value.  
Default is `2`.

**Returns:**  
The corrected working DataFrame.

---

### `fix_sales_value_by_neighbor_mean(group_cols=("Product", "Region"), target_col="Total_Price")`

Fills missing values in a target column using the mean of the previous and next values inside each group.

**Parameters:**

#### `group_cols`
Columns used to define the grouping.  
Default is `("Product", "Region")`.

#### `target_col`
Column to be corrected.  
Default is `"Total_Price"`.

**Returns:**  
The corrected working DataFrame.

**Important notes:**

- the method sorts the data by group columns and `Order_Date`
- it requires a valid temporal order
- it is useful for local interpolation inside a product-region sequence

---

### `apply_all_fixes(recalc_total_price: bool = True)`

Applies the main correction methods in sequence.

**Parameters:**

#### `recalc_total_price`
If `True`, also recalculates `Total_Price`.  
Default is `True`.

**Returns:**  
The corrected working DataFrame.

**Fix sequence applied:**

1. fill missing quantity
2. fix negative quantity
3. fill missing discount
4. fix negative discount
5. clip discount range
6. apply B2C discount rule
7. parse order date
8. remove duplicates
9. optionally recalculate total price

---

## Reporting Methods

### `generate_executive_summary() -> str`

Generates a text summary of the data quality analysis.

**Returns:**  
A formatted string with:

- number of evaluated rules
- number of rules with issues
- total issue count
- list of high-severity issues

**Behavior:**  
If the report does not exist yet, the method automatically runs the full analysis first.

---

### `get_clean_dataframe() -> pd.DataFrame`

Returns a copy of the current working DataFrame.

**Returns:**  
A pandas DataFrame.

**Use case:**  
Useful after applying corrections, when you want to continue the pipeline safely.

---

### `reset_to_original()`

Restores the working DataFrame to the original input state.

**Returns:**  
Nothing explicitly, but resets:

- `self.df`
- `self.report`

**Use case:**  
Useful when you want to discard all applied fixes and restart the analysis.

---

## Example Usage

```python
from data_quality import DataQuality

dq = DataQuality(df)

report = dq.run_full_analysis()
print(report)

print(dq.generate_executive_summary())

dq.apply_all_fixes()
df_clean = dq.get_clean_dataframe()
```

This is the most common workflow:

1. create the class
2. run the analysis
3. inspect the report
4. apply the fixes
5. export or continue with the cleaned dataset

---

## Expected Columns

This class expects the DataFrame to contain at least these columns:

- `Quantity`
- `Discount`
- `Customer_Type`
- `Order_Date`
- `Unit_Price`
- `Total_Price`
- `Product`
- `Region`


