# CSVLoader

## Overview

`CSVLoader` is a simple utility class used to load CSV files into pandas DataFrames.

It provides a small and reusable interface for reading CSV data from a folder and file name combination.

This class is useful in lightweight pipelines, notebooks, and small repository-style data loading steps.

The descriptions below are based on the uploaded source file. fileciteturn7file1

---

## Main Purpose

The main goal of `CSVLoader` is to simplify CSV loading by:

- combining a folder path and file name
- reading the CSV file into a DataFrame
- keeping the loading logic reusable in the project

---

## Class: `CSVLoader`

## Public Methods

### `create_dataframe(file_path, csv_file)`

Read a CSV file and return a pandas DataFrame.

The method joins the folder path and the file name using `Path`, then loads the file with `pd.read_csv()`.

#### Parameters

- `file_path`: Base folder path.
- `csv_file`: CSV file name.

#### Returns

- `pd.DataFrame`: Loaded CSV data.

---

## Protected Method

### `_extract_features_from_csvfile(file_path, csv_file)`

Read a CSV file and return a pandas DataFrame.

In the current implementation, this method directly calls `pd.read_csv(file_path)` and returns the loaded data. fileciteturn7file1

#### Parameters

- `file_path`: Path to the CSV file.
- `csv_file`: CSV file name parameter declared in the method signature.

#### Returns

- `pd.DataFrame`: Loaded CSV data.

#### Note

This protected method is not used inside `create_dataframe()` in the current version of the class. fileciteturn7file1

---

## Example Usage

```python
from csv_loader import CSVLoader
from pathlib import Path

loader = CSVLoader()

df = loader.create_dataframe(
    file_path=Path("data/raw"),
    csv_file="sales.csv"
)

print(df.head())
```

---

## Typical Workflow

A common workflow with this class is:

1. define the base folder
2. define the CSV file name
3. load the file into a pandas DataFrame
4. continue with validation, transformation, or modeling

---

## Notes

- The class is intentionally very small.
- It is useful when the project wants a simple loader abstraction.
- It relies on `pandas.read_csv()` for the actual file reading.
- It works well in notebook-based and pipeline-based projects.

---

## Important Considerations

### Path handling

The `create_dataframe()` method uses:

```python
arquivo = Path(file_path / csv_file)
```

This means `file_path` is expected to behave like a `Path` object or support path joining in a compatible way. fileciteturn7file1

### Minimal abstraction

This class is a lightweight wrapper around `pd.read_csv()`.  
Its main value is code organization and reuse, not advanced CSV parsing options.

---

## Summary

`CSVLoader` is a small helper class for reading CSV files into pandas DataFrames. It keeps the loading logic simple, reusable, and easy to integrate into data processing pipelines.
