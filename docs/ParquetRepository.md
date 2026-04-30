# ParquetRepository

## Overview

`ParquetRepository` is a simple utility class used to save, load, check, and delete Parquet files inside a base folder.

It helps organize file access in a cleaner and more reusable way, especially in data pipelines that work with intermediate datasets such as:

- raw data
- processed data
- feature tables
- prediction outputs
- reports saved as tables

This class is useful when a project needs a small repository layer for Parquet storage.

---

## Main Purpose

The class helps standardize file operations for Parquet datasets by:

- defining a base directory
- creating full file paths from relative paths
- saving DataFrames as Parquet files
- loading DataFrames from Parquet files
- checking if a file exists
- deleting saved files

This makes notebook and pipeline code easier to read and maintain.

---

## Class: `ParquetRepository`

## Constructor

```python
ParquetRepository(base_path: str = "data")
```

### Parameters

- `base_path` (`str`, optional): Base directory used to store and read Parquet files. Default is `"data"`.

### Main Attribute

- `base_path`: Base folder stored as a `Path` object.

---

## Public Methods

### `set_base_path(base_path: str = "data")`

Update the base directory used by the repository.

This method is useful when the project needs to switch from one storage folder to another, for example:

- from `data/raw`
- to `data/processed`
- or to a temporary test folder

#### Parameters

- `base_path` (`str`, optional): New base directory.

#### Returns

- `None`

---

### `save(df: pd.DataFrame, relative_path: str, compression: str = "snappy", index: bool = False, overwrite: bool = True) -> None`

Save a pandas DataFrame as a Parquet file.

The method automatically creates the parent folders if they do not exist.

#### Parameters

- `df` (`pd.DataFrame`): DataFrame to save.
- `relative_path` (`str`): Relative file path inside the base directory.
- `compression` (`str`, optional): Compression type used in the Parquet file. Default is `"snappy"`.
- `index` (`bool`, optional): If `True`, save the DataFrame index. Default is `False`.
- `overwrite` (`bool`, optional): If `False`, raise an error when the target file already exists. Default is `True`.

#### Returns

- `None`

#### Raises

- `FileExistsError`: If the target file already exists and `overwrite=False`.

#### Behavior

- builds the full path from `base_path` and `relative_path`
- creates missing parent folders automatically
- saves the DataFrame with `to_parquet()`
- prints a success message after saving

---

### `load(relative_path: str, columns: list = None) -> pd.DataFrame`

Load a Parquet file as a pandas DataFrame.

#### Parameters

- `relative_path` (`str`): Relative file path inside the base directory.
- `columns` (`list`, optional): Optional list of columns to load. Default is `None`, which loads all columns.

#### Returns

- `pd.DataFrame`: Loaded DataFrame.

#### Raises

- `FileNotFoundError`: If the target file does not exist.

#### Behavior

- builds the full path from `base_path` and `relative_path`
- loads the file with `pd.read_parquet()`
- prints a success message after loading

---

### `exists(relative_path: str) -> bool`

Check if a file exists inside the repository base directory.

#### Parameters

- `relative_path` (`str`): Relative file path inside the base directory.

#### Returns

- `bool`: `True` if the file exists, otherwise `False`.

---

### `delete(relative_path: str) -> None`

Delete a file from the repository base directory.

If the file exists, it is removed.  
If the file does not exist, the method prints a warning message.

#### Parameters

- `relative_path` (`str`): Relative file path inside the base directory.

#### Returns

- `None`

---

## Internal Method

### `_get_full_path(relative_path: str) -> Path`

Build the full file path using the repository base directory and the relative path.

This method is internal, but it is important for understanding how the class works.

#### Parameters

- `relative_path` (`str`): Relative file path inside the base directory.

#### Returns

- `Path`: Full file path.

---

## Example Usage

```python
from parquet_repository import ParquetRepository

repo = ParquetRepository(base_path="data")

repo.save(
    df=df_processed,
    relative_path="processed/processed.parquet"
)

df_loaded = repo.load("processed/processed.parquet")

print(df_loaded.head())

print(repo.exists("processed/processed.parquet"))

repo.delete("processed/processed.parquet")
```

---

## Example in a Pipeline

A common project flow can be:

1. load a raw CSV file
2. clean the data
3. save the processed result as Parquet
4. load the Parquet file in the next notebook or pipeline step
5. continue with feature engineering or modeling

Example:

```python
repo = ParquetRepository(base_path="data")

repo.save(df_clean, "processed/clean_sales.parquet")
df_features_input = repo.load("processed/clean_sales.parquet")
```

---

## Notes

- The class is intentionally simple and focused.
- It works with pandas DataFrames and Parquet files.
- It uses `Path` from `pathlib` for path handling.
- It is useful for notebook-based projects and modular pipelines.
- It helps keep file operations consistent across the project.

---

## Important Considerations

### Parquet support

The class depends on Parquet support in pandas.  
In the uploaded file, `pyarrow` is imported, which indicates that the environment is expected to support Parquet operations. fileciteturn6file0

### Relative path usage

The methods work with relative paths inside the configured `base_path`.  
This makes it easier to move the project to another environment without changing many hardcoded file paths.

### Overwrite control

The `save()` method includes an `overwrite` parameter.  
This is useful when the project needs protection against accidental file replacement.

---

## Summary

`ParquetRepository` is a practical utility class for projects that store intermediate datasets as Parquet files. It keeps save, load, existence check, and delete operations in one place, which makes data pipelines cleaner and easier to maintain.
