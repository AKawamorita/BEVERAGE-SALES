# TextLoader

## Overview

`TextLoader` is a utility class used to read bearing signal files and convert them into a structured pandas DataFrame.

It is designed for datasets where each file contains sensor signal values and the file name may represent a timestamp.

The class supports:

- reading individual bearing files
- extracting signal-based features from multiple files
- creating one DataFrame for a full run
- calculating a simple Remaining Useful Life (`RUL`) target
- optionally using timestamps from file names

This class is useful in predictive maintenance, condition monitoring, and time series anomaly projects.

The descriptions below are based on the uploaded source file. fileciteturn7file0

---

## Main Purpose

The main goal of `TextLoader` is to transform many raw bearing signal files into one structured dataset that can be used for:

- exploratory analysis
- feature engineering
- anomaly detection
- health index modeling
- Remaining Useful Life estimation

---

## Class: `TextLoader`

## Public Methods

### `read_bearing_file(file_path)`

Read one bearing signal file as a pandas DataFrame.

The method uses whitespace-separated values and does not expect a header row.

#### Parameters

- `file_path`: Path to the signal file.

#### Returns

- `pd.DataFrame`: DataFrame containing the raw sensor signal values.

---

### `create_dataframe_old(run_folder, run_id, max_sensors=4)`

Create a DataFrame for one run using the original loading logic.

This method:

1. reads all files inside the run folder
2. extracts features from each file
3. creates one row per file
4. computes the `RUL` column using the largest `time_index`

#### Parameters

- `run_folder`: Folder containing the files for one run.
- `run_id`: Run identifier used in the output dataset.
- `max_sensors` (`int`, optional): Maximum number of sensor columns used from each file. Default is `4`.

#### Returns

- `pd.DataFrame`: DataFrame with extracted features and `RUL`.

#### Output columns

The resulting DataFrame includes, at minimum:

- `run_id`
- `time_index`
- `file_name`
- extracted signal feature columns
- `RUL`

---

### `create_dataframe(run_folder, run_id, max_sensors=4)`

Create a DataFrame for one run using file-name timestamps when available.

This is the main public loading method in the class.

It performs the following steps:

1. reads all files from the run folder
2. tries to convert each file name into a real timestamp using the format:

```python
%Y.%m.%d.%H.%M.%S
```

3. extracts signal features from each file
4. creates one row per file
5. converts the timestamp column to datetime
6. sets `timestamp` as the DataFrame index
7. computes the `RUL` column

If a file name cannot be parsed as a timestamp, the method stores `None` for that row before conversion.

#### Parameters

- `run_folder`: Folder containing the files for one run.
- `run_id`: Run identifier used in the output dataset.
- `max_sensors` (`int`, optional): Maximum number of sensor columns used from each file. Default is `4`.

#### Returns

- `pd.DataFrame`: DataFrame with extracted features, timestamp index, and `RUL`.

#### Output columns

The resulting DataFrame includes, at minimum:

- `run_id`
- `time_index`
- `file_name`
- `timestamp`
- extracted signal feature columns
- `RUL`

#### Important behavior

If the `timestamp` column exists, the method converts it to datetime and sets it as the DataFrame index. fileciteturn7file0

---

## Protected Method

### `_extract_features_from_file(file_path, run_id, time_index, max_sensors=4)`

Extract signal features from one file and return them as a dictionary.

This method:

- reads the file with `read_bearing_file()`
- creates metadata fields such as `run_id`, `time_index`, and `file_name`
- applies `ft.extract_signal_features(...)` to each sensor column
- combines all extracted features into one row dictionary

#### Parameters

- `file_path`: Path to the input file.
- `run_id`: Run identifier.
- `time_index`: Sequential position of the file inside the run.
- `max_sensors` (`int`, optional): Maximum number of sensors processed.

#### Returns

- `dict`: Dictionary with metadata and extracted features.

---

## Dependencies and Expected Behavior

The class depends on an external feature extraction function:

```python
ft.extract_signal_features(signal, prefix=sensor_prefix)
```

This means the project must provide that function in `src.features`. fileciteturn7file0

The class also assumes that each input file contains sensor values organized by columns.

---

## Example Usage

```python
from text_loader import TextLoader

loader = TextLoader()

df_run = loader.create_dataframe(
    run_folder="data/1st_test/1st_test",
    run_id="run_1",
    max_sensors=4
)

print(df_run.head())
print(df_run.columns)
```

---

## Typical Workflow

A common workflow with this class is:

1. select one run folder
2. load all bearing files from that folder
3. extract signal features from each file
4. build a structured DataFrame
5. use the output for anomaly detection or RUL modeling

---

## Notes

- The class is designed for bearing datasets stored as many text-like files.
- It supports a maximum sensor limit with `max_sensors`.
- It adds a simple `RUL` calculation based on file order.
- It can use timestamps extracted from file names.
- It is useful for time series and predictive maintenance pipelines.

---

## Important Considerations

### File naming format

The main `create_dataframe()` method expects file names in the format:

```python
YYYY.MM.DD.HH.MM.SS
```

If the file name does not follow this format, the timestamp becomes missing for that file. fileciteturn7file0

### External feature extractor

Feature extraction is not implemented directly inside the class.  
It depends on `src.features.extract_signal_features(...)`. fileciteturn7file0

### RUL calculation

The `RUL` column is calculated as:

```python
max_time_index - current_time_index
```

This gives a simple decreasing target across the files of one run. fileciteturn7file0

---

## Summary

`TextLoader` is a practical class for turning many bearing signal files into a structured feature dataset. It helps connect raw sensor files with later modeling steps such as anomaly detection, condition monitoring, and Remaining Useful Life analysis.
