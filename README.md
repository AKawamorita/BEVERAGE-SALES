# Beverage Sales Analytics and Hybrid ML Pipeline

## Project Overview

This project explores a **Beverage Sales** dataset and builds a structured analytics and machine learning workflow focused on:

- sales intelligence
- data quality validation
- time-based feature engineering
- anomaly detection
- baseline modeling
- hybrid modeling
- performance benchmarking

---

## Business Motivation

Sales data often contains:

- seasonality
- regional demand differences
- product-level variation
- unusual peaks and drops
- behavior changes over time

Because of that, this project investigates both:

1. **normal sales patterns**
2. **unusual sales behavior through anomaly detection**

The main idea is to test whether anomaly-related signals can improve supervised model performance when compared to a simpler baseline approach.

---

## Main Objectives

This project was created to answer questions such as:

- How do beverage sales behave over time?
- Can we detect unusual demand spikes and drops?
- Which time-based features help describe sales dynamics?
- Does anomaly information add value to a supervised model?
- How does a hybrid model compare against a baseline model?

---

## Notebook Execution Order

The project was organized as a notebook pipeline. The recommended execution order is:

1. `data_loading.ipynb`
2. `data_quality.ipynb`
3. `feature_engineering.ipynb`
4. `anomaly_detection.ipynb`
5. `BaseLine.ipynb`
6. `HybridModel.ipynb`
7. `performance_benchmarking.ipynb`

---

## Notebook Summary

### 1. `data_loading.ipynb`

#### Purpose
Loads the raw dataset, standardizes columns, converts data types, and prepares the base data for the next stages.

#### Main tasks
- load raw CSV data
- inspect schema and columns
- standardize structure
- convert date fields
- save processed data

#### Output
- cleaned dataset
- processed file ready for downstream analysis

---

### 2. `data_quality.ipynb`

#### Purpose
Validates whether the processed dataset is reliable enough for analysis and modeling.

#### Main tasks
- missing value checks
- invalid or inconsistent record detection
- business-rule validation
- quality summary generation

#### Output
- quality report
- validated dataset for feature engineering

---

### 3. `feature_engineering.ipynb`

#### Purpose
Transforms raw sales data into time-based and business-oriented features for analysis and machine learning.

#### Main tasks
- aggregate sales information
- create rolling window statistics
- generate lag-based features
- build ratio-based features
- prepare model-ready inputs

#### Example features
- `quantity_sum`
- `total_price_sum`
- `unit_price_mean`
- `discount_mean`
- `order_count`
- `customer_count`
- `avg_ticket`
- rolling means
- rolling standard deviations
- historical comparison metrics

#### Output
- feature-enriched dataset

---

### 4. `anomaly_detection.ipynb`

#### Purpose
Detects unusual sales behavior using anomaly detection logic.

#### Main tasks
- load engineered features
- train or load anomaly detection model
- generate anomaly scores
- generate anomaly flags
- save anomaly-enriched outputs

#### Output
- anomaly scores
- anomaly flags
- enriched dataset for downstream modeling

---

### 5. `BaseLine.ipynb`

#### Purpose
Builds the baseline supervised model used as the main reference point in the project.

#### Main tasks
- load modeling features
- prepare train / validation / test logic
- train the baseline model
- evaluate predictive performance
- generate benchmark metrics

#### Output
- trained baseline model
- baseline metrics
- reference results for comparison

---

### 6. `HybridModel.ipynb`

#### Purpose
Builds the hybrid solution by combining anomaly-related information with supervised learning.

#### Main tasks
- load engineered features
- include anomaly outputs as additional signals
- train the hybrid model
- evaluate predictive performance
- compare hybrid results against the baseline

#### Output
- trained hybrid model
- hybrid model metrics
- comparison-ready outputs

---

### 7. `performance_benchmarking.ipynb`

#### Purpose
Consolidates results and compares the final performance of the modeling approaches.

#### Main tasks
- gather baseline results
- gather hybrid model results
- compare metrics
- visualize model differences
- summarize performance trade-offs

#### Output
- benchmark tables
- comparison plots
- final performance summary

---

## Workflow Logic

The project follows this logic:

**raw data -> data quality -> feature engineering -> anomaly detection -> baseline model -> hybrid model -> performance benchmarking**

---

## Visual Analysis

This project also includes visual analysis to improve interpretability and business understanding.

### Boxplot

Used to summarize the distribution of `Quantity` with focus on:

- median
- quartiles
- spread
- possible outliers

### Violin Plot

Used when a richer view of the distribution is needed.

Why it is useful:

- shows density shape
- highlights concentration zones
- makes long-tail behavior easier to interpret

### Why `Quantity` was used instead of `quantity_sum` in distribution plots

For distribution analysis, the raw `Quantity` field is more appropriate because it preserves the natural variability of the observations.

Using `quantity_sum` in boxplots or violin plots could distort interpretation because it is already an aggregated metric.

### Outlier Plot by Period

This chart was used to detect unusual **daily total demand** over time.

For this use case, aggregation makes sense because the goal is to identify abnormal daily behavior rather than the distribution of individual records.

---

## Technical Design Choices

### Parameterized filters

Visualization functions were designed with reusable filters such as:

- `Category`
- `Region`

Example:

```python
filters = {
    "Category": ["Soft Drinks", "Water", "Juices"],
    "Region": ["Baden-Württemberg"]
}
```

This improves:

- code reuse
- flexibility across business slices
- maintainability
- future reuse in dashboards or APIs

### Hybrid modeling idea

A central idea of the project is that anomaly-related information can complement supervised learning.

This reflects a practical machine learning engineering mindset:

- use unsupervised signals to enrich structured data
- compare against a simpler baseline
- validate whether extra complexity adds real value

---

## Project Strengths

This project demonstrates:

- structured notebook pipeline
- data preparation discipline
- data quality validation
- time-based feature engineering
- anomaly detection for business data
- baseline vs hybrid model comparison
- reusable Python components
- visual analysis for business interpretation
- portfolio-ready technical documentation

---

## Skills Demonstrated

### Core technical skills
- Python
- Pandas
- NumPy
- Matplotlib
- Scikit-learn
- Feature Engineering
- Exploratory Data Analysis (EDA)
- Data Visualization
- Anomaly Detection
- Supervised Machine Learning
- Model Benchmarking
- Time-Series Feature Engineering
- Business Data Analysis

### ML and analytics skills
- baseline model design
- hybrid modeling strategy
- outlier analysis
- distribution analysis
- rolling window features
- lag-based reasoning
- anomaly-aware modeling
- evaluation and comparison of models

### Engineering and project skills
- notebook pipeline structuring
- reusable code design
- parameterized visual components
- technical documentation
- GitHub-ready project organization
- analytical storytelling

---

## Suggested LinkedIn Skills

Based on what was actually built, these are the strongest and most honest skills to highlight on LinkedIn.

### High-impact skills
- Machine Learning
- Python
- Feature Engineering
- Anomaly Detection
- Time Series Analysis
- Predictive Modeling
- Scikit-learn
- Pandas
- Data Analysis
- Data Visualization

### Strong supporting skills
- Exploratory Data Analysis
- Statistical Data Analysis
- Model Evaluation
- Business Analytics
- Sales Analytics
- Outlier Detection
- Data Quality
- Technical Documentation

### Good options for an ML Engineer positioning
- Machine Learning Engineering
- ML Pipelines
- Model Benchmarking
- Applied Machine Learning
- Reusable Data Components

---

## Suggested LinkedIn Skill Order

A strong order could be:

1. Machine Learning
2. Python
3. Feature Engineering
4. Anomaly Detection
5. Time Series Analysis
6. Predictive Modeling
7. Scikit-learn
8. Pandas
9. Data Analysis
10. Data Visualization
11. Exploratory Data Analysis
12. Statistical Data Analysis
13. Business Analytics
14. Model Evaluation
15. Technical Documentation

---

## Final Notes

This project is valuable for a portfolio because it shows more than model training.

It demonstrates:

- data preparation
- practical feature engineering
- anomaly reasoning
- structured experimentation
- model comparison
- business interpretation
- technical communication


