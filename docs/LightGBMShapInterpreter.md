# LightGBMShapInterpreter

## Overview

`LightGBMShapInterpreter` is a helper class used to explain a trained LightGBM model with SHAP.

It is designed for models stored inside a scikit-learn `Pipeline` that contains:

- a fitted preprocessing step named `preprocessor`
- a fitted model step named `model`

The class transforms the original input data, creates a SHAP explainer, computes SHAP values, plots a SHAP summary, and returns feature importance based on mean absolute SHAP values.

This class is useful for explainability in machine learning projects that use preprocessing pipelines before a LightGBM model.

The descriptions below are based on the uploaded source file. fileciteturn8file0

---

## Main Purpose

The main goal of `LightGBMShapInterpreter` is to make model interpretation easier after training.

It helps answer questions such as:

- which features most influenced the model
- how important each transformed feature is
- how the pipeline preprocessing affects the final model explanation

This is especially useful in tabular machine learning projects where the model uses:

- imputation
- one-hot encoding
- transformed numeric and categorical features

---

## Class: `LightGBMShapInterpreter`

## Constructor

```python
LightGBMShapInterpreter(pipeline: Pipeline)
```

### Parameters

- `pipeline` (`Pipeline`): Trained scikit-learn pipeline that must contain:
  - `preprocessor`
  - `model`

### Validation Rules

The constructor validates the following conditions:

- the pipeline cannot be `None`
- the object must be a scikit-learn `Pipeline`
- the pipeline must contain a `preprocessor` step
- the pipeline must contain a `model` step

If any of these conditions are not satisfied, the constructor raises a `ValueError`. fileciteturn8file0

### Main Attributes

- `pipeline`: Full trained scikit-learn pipeline.
- `preprocessor`: Fitted preprocessing step extracted from the pipeline.
- `model`: Fitted model step extracted from the pipeline.
- `explainer`: Internal SHAP explainer object. It starts as `None` and is created when needed.

---

## Public Methods

### `transform_features(df: pd.DataFrame) -> pd.DataFrame`

Transform the original input DataFrame using the fitted preprocessor.

This method is useful because SHAP explanations must be calculated on the transformed feature space used by the model.

The method also tries to recover feature names from the preprocessor.

If feature names are not available, it creates generic names such as:

- `feature_0`
- `feature_1`
- `feature_2`

If the transformed output is sparse, the method converts it to a dense array before building the returned DataFrame. fileciteturn8file0

#### Parameters

- `df` (`pd.DataFrame`): Original DataFrame with raw input columns.

#### Returns

- `pd.DataFrame`: Transformed DataFrame ready for SHAP analysis.

---

### `create_explainer()`

Create a SHAP `TreeExplainer` for the internal LightGBM model.

This method stores the explainer in the class and also returns it.

#### Returns

- `shap.TreeExplainer`: Configured SHAP explainer.

---

### `compute_shap_values(df: pd.DataFrame)`

Compute SHAP values for the given input DataFrame.

If the explainer does not exist yet, the method creates it automatically.

The method returns both:

- the SHAP values
- the transformed feature DataFrame used in the explanation

This is useful because the transformed features are the exact input used by the model after preprocessing.

#### Parameters

- `df` (`pd.DataFrame`): Original DataFrame with raw input columns.

#### Returns

- `tuple`:
  - `shap_values`
  - `X_transformed_df`

---

### `plot_summary(df: pd.DataFrame, plot_type: str = "dot") -> None`

Create a SHAP summary plot.

This method computes SHAP values and then calls `shap.summary_plot(...)`.

Common plot types include:

- `"dot"`
- `"bar"`

#### Parameters

- `df` (`pd.DataFrame`): Original DataFrame with raw input columns.
- `plot_type` (`str`, optional): Type of SHAP summary plot. Default is `"dot"`.

#### Returns

- `None`

---

### `get_feature_importance(df: pd.DataFrame) -> pd.DataFrame`

Return a DataFrame with mean absolute SHAP importance per transformed feature.

This method computes SHAP values, takes the absolute value, calculates the mean importance by feature, and sorts the result in descending order.

The output contains:

- `feature`
- `mean_abs_shap`

#### Parameters

- `df` (`pd.DataFrame`): Original DataFrame with raw input columns.

#### Returns

- `pd.DataFrame`: Feature importance table sorted from highest to lowest importance.

#### Important behavior

If `shap_values` is returned as a list, the method uses the first element before calculating the importance values. fileciteturn8file0

---

## Typical Workflow

A common workflow with this class is:

1. train a machine learning pipeline
2. pass the trained pipeline to `LightGBMShapInterpreter`
3. transform the original features
4. compute SHAP values
5. create a SHAP summary plot
6. inspect the feature importance table

---

## Example Usage

```python
from LightGBMShapInterpreter import LightGBMShapInterpreter

interpreter = LightGBMShapInterpreter(trained_pipeline)

shap_values, X_transformed = interpreter.compute_shap_values(df_test)

importance_df = interpreter.get_feature_importance(df_test)
print(importance_df.head())

interpreter.plot_summary(df_test, plot_type="bar")
```

---

## Expected Pipeline Structure

The class expects a trained scikit-learn `Pipeline` with these step names:

```python
pipeline.named_steps["preprocessor"]
pipeline.named_steps["model"]
```

This means the training pipeline should already be fitted before this interpreter is used. fileciteturn8file0

---

## Notes

- This class is focused on explainability, not model training.
- It works with a fitted preprocessing pipeline and a fitted LightGBM model.
- It helps interpret the transformed feature space used internally by the model.
- It is useful in projects that need explainable machine learning results.

---

## Important Considerations

### Transformed feature space

The SHAP analysis is computed after preprocessing.  
This means the feature names and values may be different from the original raw dataset, especially when categorical variables are one-hot encoded. fileciteturn8file0

### Sparse to dense conversion

If the transformed matrix is sparse, the class converts it to a dense array before building a pandas DataFrame.  
This is practical for explanation, but it may use more memory for very large datasets. fileciteturn8file0

### SHAP output format

Depending on the SHAP version and model behavior, SHAP values may come as a list or as a numeric array.  
The class already handles this case when calculating mean absolute feature importance. fileciteturn8file0

---

## Summary

`LightGBMShapInterpreter` is a practical helper class for explaining trained LightGBM pipelines with SHAP. It transforms the input data, computes SHAP values, creates summary plots, and returns a feature importance table that supports model interpretation and explainability.
