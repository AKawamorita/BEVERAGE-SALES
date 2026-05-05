# Beverage Sales Machine Learning Project  
## Sales forecasting supported by anomaly detection

## 1. Project overview

This project aims to analyze historical beverage sales and build a model capable of **forecasting demand** with support from **signals of unusual behavior**.

- Dataset : https://www.kaggle.com/datasets/sebastianwillmann/beverage-sales

The dataset has a significant volume of more than 540,000 records.  
- Operational Stability: In normal records, the 0.32% RMSE improvement and the reduction in Max AE (Maximum Error) from 158 to 144 are valuable metrics for logistics. In a large-scale operation (hundreds of thousands of units), reducing the maximum error by 14 units in critical cases helps avoid stockouts or unnecessary excess inventory in fast-moving products.  
- Exception Management: 5,571 anomalous records were identified. Although they represent only about 1% of the total, these are the cases where the operational cost of error is higher (for example: special orders or unexpected spikes). 

The main idea was to combine:

- a **supervised** model for forecasting the quantity sold;
- an **unsupervised** model to identify anomalous or rare patterns in the data.

In practice, the project's purpose was to check whether adding anomaly information to the pipeline could help the main model deal better with unusual situations, such as uncommon sales fluctuations, atypical behavior in price, discount, or volume.

#### 1.1 ⚙️The project follows this logic:
**raw data -> data quality -> feature engineering -> anomaly detection -> baseline model -> hybrid model -> performance benchmarking**

#### 1.2 ⚙️Execution order
📝01_data_loading.ipynb -> 📝02_data_quality.ipynb -> 📝03_feature_engineering.ipynb -> 📝04_anomaly_detection.ipynb -> 
📝05_BaseLine.ipynb -> 📝06_HybridModel.ipynb -> 📝07_Interpretability_SHAP_Analysis.ipynb -> 📝08_performance_benchmarking.ipynb 

#### 1.3 Suggested execution order:

1. `01_data_loading.ipynb`
2. `02_data_quality.ipynb`
3. `03_feature_engineering.ipynb`
4. `04_anomaly_detection.ipynb`
5. `05_BaseLine.ipynb`
6. `06_HybridModel.ipynb`
7. `07_Interpretability_SHAP_Analysis.ipynb`
8. `08_performance_benchmarking.ipynb`

#### 1.4 Libraries used
- All libraries used are listed in:
  🧾requirements.txt

#### 1.5 Project split
The project is divided into 2 parts
- The Notebook part, understanding the problem, the dataset, training, and result analysis
- The part focused on the application, Flask, REST API, and a suggested model application
![Model illustration](img/InfoGraphic50.png)

---

## 2. Business problem

Even in a synthetic or semi-synthetic dataset, the problem can be interpreted as a real business scenario:

- forecast future sales by product and region;
- capture recent temporal patterns with sliding windows;
- improve model robustness in less stable periods;
- support decisions related to stock planning, replenishment, and demand behavior analysis.

In practical terms, the project tries to answer questions such as:

- **how much can be sold in the next period?**
- **is the current behavior within the historical pattern or different from normal?**
- **can a model that knows anomaly signals make fewer errors?**

---

## 3. Strategy adopted

The solution was designed in two parts:

### 3.1 Baseline model
A supervised regression model was created as a baseline, using only business variables and time-series variables derived from history, without explicit support from anomalies.

This model represents the main reference for comparison.

### 3.2 Model supported by anomaly detection
Then, an anomaly detection layer was added to the pipeline, generating variables such as:

- `anomaly_flag`
- `anomaly_score`

These variables were then used together with the other features in the final supervised model.

The goal was not to replace the forecasting model, but to **enrich it with an extra signal** about unusual contexts. For this reason, both models are "the same", meaning both are LightGBM: one without anomaly detection support (Baseline) and the other with anomaly detection support (Isolation Forest + LightGBM)


## 3.3 Sliding Window analysis in the Beverage Sales project

In the **Beverage Sales** project, the use of **Sliding Window** was one of the most important parts of the modeling.  
The main idea was to transform historical sales data into **useful temporal signals for the model**, allowing it to see not only the current value, but also the **recent demand behavior**.

![Model illustration](img/Sliding_Window.png)

Instead of using only raw columns, the project created variables that summarize the closest history of each business combination, helping the model answer questions such as:

- is this sale above or below the recent pattern?
- was there an acceleration or slowdown in demand?
- do recent price or discount changes seem to influence volume?
- does this point in time look normal or unusual for that product and region?

In practice, Sliding Window was used to create a **short-term and medium-term** view of the series, enriching the data used by the **LightGBM Regressor** and, in the hybrid model, also by the anomaly signal generated by the **Isolation Forest**.

---

## 3.3.1 What is Sliding Window

Sliding Window, or **moving window**, is a technique used in time series to calculate statistics over a moving set of past observations.

Instead of looking at the whole history at once, the algorithm looks only at a recent **time window**, for example:

- last 7 days
- last 14 days
- last 30 days

This window “slides” over time.  
For each new date, the system recalculates statistics based only on the available previous history.

This is useful because, in demand and sales problems, recent behavior usually carries very important signals, such as:

- local trend
- pace changes
- recent increase or decrease
- stability or volatility
- price and discount effects

---

## 3.3.2 Why Sliding Window was important in this project

The Beverage Sales dataset has a temporal nature.  
Even though it is treated as a tabular modeling problem, sales behavior changes over time due to factors such as:

- seasonality
- promotions
- regional variations
- differences between products
- recent demand fluctuations

If the model used only static columns or single values, it would lose much of this context.

Sliding Window solved this by creating features that represent the **recent history of each local series**, making the modeling richer without needing to move to a more complex architecture, such as LSTM or Transformer.

In other words, it made it possible to capture an important part of the temporal dynamics using a simpler, more robust, and easier-to-explain pipeline.

---

## 3.3.3 How Sliding Window worked in the project

In the project, Sliding Window was applied to aggregated sales metrics, respecting the temporal order and the business granularity.

The general logic was:

1. organize the data by time;
2. separate the series by business context, such as **Product** and **Region**;
3. calculate moving statistics on columns of interest;
4. use only past information to build the new features;
5. feed the model with these derived variables.

This process allowed each row in the dataset to carry a small summary of the recent behavior of that series.

---

## 3.3.4 Time window and prevention of data leakage

A very important point was the use of **shift(1)** before the moving calculations.

This means that, when calculating a 7-day average for a given date, the project **did not include the current day's own value inside the window**.  
The feature was built only from previous information.

This detail is essential to avoid **data leakage** or **temporal leakage**.

---

## 3.3.5 Main windows used

The project worked with windows such as:

- **7 days**: short view, useful to capture recent behavior
- **14 days**: intermediate view, used in some variables such as discount
- **30 days**: more stable view, useful to compare the present against a broader pattern

Each window had a different role.

### 7-day window
It was important to capture fast-changing signals, for example:

- recent volume increase
- recent sales drop
- local quantity behavior
- recent change in ticket size or price

### 14-day window
It was useful for variables that may have moderate fluctuations, such as average discounts.

### 30-day window
It was useful to form a more stable reference and less sensitive to very short-term noise.

---

## 3.3.6 Features created with Sliding Window

The project created several temporal features based on moving windows.  
Among them, the following stand out:

### Quantity
- `quantity_sum_mean_7d`
- `quantity_sum_std_7d`
- `quantity_sum_sum_7d`

These variables help the model understand:

- recent average sales
- recent variability
- recent accumulated volume

### Total price
- `total_price_sum_mean_7d`
- `total_price_sum_std_7d`
- `total_price_sum_mean_30d`

These features help show the recent behavior of the total value moved.

### Unit price
- `unit_price_mean_mean_7d`

It helps capture whether there was a change in the recent average price pattern.

### Discount
- `discount_mean_mean_14d`

It helps represent the recent discount intensity.

### Ratios and comparisons with history
- `quantity_vs_mean_7d`
- `total_price_vs_mean_30d`
- `quantity_pct_vs_mean_7d`

These were especially relevant because they transform history into a **relative signal**, not just an absolute one.

For example:

- selling 20 units can be a lot or a little, depending on the pattern of that product/region;
- an isolated absolute value does not tell the whole story;
- a ratio against the recent average gives the model more context.

---

## 3.3.7 Why relative features were so important

In sales projects, the absolute value alone does not always say much.

Example:

- for a low-volume product, selling 20 can be a spike;
- for a high-volume product, selling 20 can be a strong drop.

The relative features created by Sliding Window help exactly with this.  
They tell the model **how the current value is positioned compared to its own recent history**.

This improves context reading and tends to increase the model's ability to detect:

- demand acceleration
- loss of sales strength
- unusual deviations
- possible anomalous behavior

---

## 3.3.8 Relationship between Sliding Window and Isolation Forest

In the project's hybrid model, Sliding Window has an even more important role, because it helped not only the supervised model, but also the anomaly detection stage.

**Isolation Forest** does not work directly with the idea of time like a classic time series model.  
It sees patterns in the feature space.

Because of this, the better the temporal representation embedded in the features, the better the anomaly detector tends to identify unusual points.

In other words:

- Sliding Window transformed history into measurable variables;
- Isolation Forest used these variables to identify out-of-pattern behavior;
- LightGBM, in the hybrid model, started receiving both temporal features and anomaly signals.

This combination was smart because each stage had a different role:

- **Sliding Window**: represents recent history
- **Isolation Forest**: marks what seems unusual
- **LightGBM**: learns to forecast using these combined signals

---

## 3.3.9 Benefits of Sliding Window in this project

- 1. Added memory to the model
Without Sliding Window, the model would see less historical context.  
With it, each row started carrying a summarized memory of the recent past.

- 2. Improved the reading of local trend
The model started having signals about recent increases, stability, or declines in sales.

- 3. Helped contextualize volume, price, and discount
Instead of looking only at the value of the day, the model could compare the present with the recent pattern.

- 4. Made the detection of unusual behavior easier
The moving features gave the Isolation Forest a better basis to identify deviations.

- 5. Kept the project interpretable
Even though it is a powerful technique, Sliding Window remains relatively easy to explain in a portfolio, interview, and documentation.

----

## 3.3.10 Business interpretation

From a business point of view, Sliding Window helped the project respond better to situations such as:

- is a product accelerating in demand?
- was there a recent drop outside the pattern?
- is the current behavior aligned with the recent average?
- do recent discounts seem to be changing the sold volume?
- are we facing a normal period or an unusual point?

This is valuable because it brings the project closer to real decisions, such as:

- demand forecasting
- sales monitoring
- performance analysis by product and region
- support for replenishment and planning
- identification of more sensitive or unstable periods

---

## 4. Feature engineering

The project uses a tabular time-series approach, focused on **sliding window** and historical aggregations.

Among the groups of variables that make sense in this context, there are:

- recent sales history;
- moving averages;
- sums and deviations in time windows;
- indicators related to recent behavior;
- price, discount, and volume signals;
- temporal variables, such as month, day of the week, and weekend;
- signaling of anomalous observations through an unsupervised model.

This approach is important because it allows the model to capture:

- seasonality;
- local behavior changes;
- recent demand acceleration or slowdown;
- instability at certain points in the series.

---

## 5. Evaluated metrics

To compare the models, the main observed metrics were:

- **MAE** (Mean Absolute Error)
- **Median AE** (median absolute error)
- **RMSE** (Root Mean Squared Error)
- maximum error
- absolute improvement
- percentage improvement

Besides the overall evaluation, the results were separated by:

- `anomaly_flag = 0` → records considered normal
- `anomaly_flag = 1` → records considered anomalous

This split is very important, because the hybrid model's gain may be small in the aggregate, but more relevant exactly where the problem is harder.

---

## 6. Results obtained

## 6.1 Results for normal records (`anomaly_flag = 0`)

![Model Comparison](img/ModelComparison45.png)

- **Number of records in the test set:** 543,389
- **Baseline MAE:** 2.692384
- **Anomaly-supported model MAE:** 2.689313
- **Absolute MAE improvement:** 0.003071
- **Percentage MAE improvement:** 0.114065%

- **Baseline RMSE:** 3.976524
- **Anomaly-supported model RMSE:** 3.963730
- **Absolute RMSE improvement:** 0.012795
- **Percentage RMSE improvement:** 0.321760%

- **Baseline Median AE:** 1.916530
- **Anomaly-supported model Median AE:** 1.910264

- **Baseline Max AE:** 158.320738
- **Anomaly-supported model Max AE:** 144.778257

The 1.81% RMSE improvement for this group indicates that the hybrid model is more resilient to demand "shocks".

### Interpretation
In records considered normal, the gain of the hybrid model was **small, but consistent**.

This suggests that, in scenarios where behavior already follows the expected pattern, the baseline model is already strong, and the anomaly layer adds only a marginal refinement.

Even so, there are positive points:

- the average error decreased;
- RMSE decreased;
- the median error decreased;
- the maximum error also decreased in a relevant way.

In other words, even when the average gain is modest, the anomaly-supported model showed slightly more stable behavior.

---

## 6.2 Results for anomalous records (`anomaly_flag = 1`)

- **Number of records:** 5,571
- **Baseline MAE:** 6.156137
- **Anomaly-supported model MAE:** 6.109850
- **Absolute MAE improvement:** 0.046286
- **Percentage MAE improvement:** 0.751875%

- **Baseline RMSE:** 12.726619
- **Anomaly-supported model RMSE:** 12.495805
- **Absolute RMSE improvement:** 0.230814
- **Percentage RMSE improvement:** 1.813630%

- **Baseline Median AE:** 2.514491
- **Anomaly-supported model Median AE:** 2.495663

- **Baseline Max AE:** 123.955175
- **Anomaly-supported model Max AE:** 124.363004

### Interpretation
In anomalous records, the gain of the hybrid model was **more noticeable**, mainly in:

- MAE
- RMSE
- median error

This is an important result, because anomalous cases are exactly the hardest ones for the traditional supervised model.

In other words:

- when behavior stays within the pattern, the gain exists, but it is small;
- when behavior moves outside the pattern, the anomaly signal helps more.

This reinforces the hypothesis that using an unsupervised model can add value as an auxiliary feature in forecasting problems.

The only point where there was no improvement was the **maximum error**, which became slightly worse in the anomaly-supported model. This shows that the solution does not solve every extreme case, but it still improved the most representative metrics of the set.

---
## 6.3 SHAP Explainability

1. SHAP Summary Plot (Global Importance)
The dot plot shows which variables most "push" the sales volume prediction (quantity_sum) up or down.

Dominance of Moving Averages: The variables numeric_quantity_vs_mean_7d and numeric_quantity_sum_mean_7d are by far the most impactful. This indicates that the model<br>
is strongly guided by recent sales behavior.  

Direct Correlation: Notice that high values (red) of these variables are on the right side of the zero axis, which means that if sales were high in the last 7 days, <br>
the prediction for the next period also tends to be high.  

Order Count: The number of orders (numeric_order_count) appears as the fourth most important factor, acting as a strong validator of volume.  

2. SHAP Dependence Plot (The role of anomaly_score)
This chart is crucial to understand how anomaly detection is influencing LightGBM.
![Dependence plot](img/Dependence_Plot.png)

Nonlinear Impact: <br>
- The SHAP value for anomaly_score stays close to zero most of the time, but shows positive peaks (impact of up to +20 on prediction)<br>
when the score is between 0.0 and 0.1.  

Underestimation correction:<br> 
The red points at the top of the chart indicate that, when the current volume is much higher than the average (high quantity_vs_mean_7d), anomaly_score helps the model "accept" <br>
this high value, increasing the prediction to reduce the underestimation error that the baseline would make.

3. Critical Cases and Waterfall Charts

![Dependence plot](img/Waterfall_plot.png)

The critical_cases table shows that the hybrid model significantly reduced the absolute error in high-volume transactions:<br>

| Index | Real Value | Baseline Pred. | Hybrid Pred. | Error Reduction |
| :--- | :--- | :--- | :--- | :--- |
| **204569** | 867.0 | 781.09 | 828.91 | 47.82 |
| **804841** | 1115.0 | 1023.37 | 1070.35 | 46.98 |

The specific values extracted from your results were:<br>
- Case 204569: Real 867.0, Baseline 781.09, and Hybrid 828.91.  
- Case 804841: Real 1115.0, Baseline 1023.37, and Hybrid 1070.35.  

Error reduction: For the first case, the improvement was approximately 47.82, and for the second, 46.98. 

## 7. Did the project achieve its goal?

## Yes, partially — and in a technically valid way.

The most honest answer is:

### What was achieved
- It was possible to build a clear baseline.
- It was possible to compare baseline versus anomaly-supported model.
- The anomaly-supported model showed **consistent improvement** in the main metrics.
- The gain was **more relevant exactly in anomalous records**, which are the hardest and most interesting from an analytical point of view.
- The project's main hypothesis was validated: **an unsupervised signal can enrich a supervised forecasting model**.

### What was not strongly achieved
- The global gain was not large.
- In normal records, the improvement was small.
- The project did not show a radical performance change.
- The maximum error in anomalous records did not improve.

---

## 8. Financial and Business Impact

Considering that the dataset includes price and discount variables:  
 - Margin Protection: By using anomaly_score as a feature, the model starts to "understand" better when a high sales volume is linked 
 to an aggressive discount or a data entry error. This helps prevent the purchasing system from repeating a high order based on an 
 anomaly that will not happen again.  
 - Reduction of Idle Inventory: The consistent improvement in MAE and Median AE, even if small in percentage terms, translates into accumulated 
 financial savings when applied across the whole product portfolio. In the long term, less average error means more efficient working capital.  

---

## 9. Technical conclusion

The results indicate that the hybrid approach is **promising**, but its impact was **moderate** in this dataset, showing a mature and realistic analysis.

In real Machine Learning problems, a new layer does not always generate spectacular gains. Very often, the value is in showing that:

- there is a well-defined hypothesis;
- there is a comparison baseline;
- there is a controlled experiment;
- the improvement was measured correctly;
- the solution helps more exactly in the hardest cases.

In this project, that is what happened.

The anomaly-supported model:

- did not revolutionize the overall performance, which was expected because it was the same LightGBM model, one only with the model and the other supported by Isolation Forest.
- but showed consistent gain;
- and showed more important gain in the subset where the problem is more complex.

---

## 10. Project limitations

Some important limitations should be recognized:

### 9.1 Dataset
The dataset used appears to have synthetic or simplified characteristics, which may limit the depth of the patterns found.

### 9.2 Low proportion of anomalies
The number of anomalous records is small compared to the normal set. This reduces the global impact of anomaly features on the overall average.

### 9.3 Incremental improvement
The gains were real, but incremental. This suggests that the baseline was already strong and that the room for further improvement was limited.

### 9.4 Scope
The project was focused on tabular forecasting supported by anomaly detection, and not on a complete production stock optimization or operational decision system.

---

## 11. Project value for portfolio

Even with modest gains, this project is relevant for a portfolio because it shows:

- use of **tabular time series**;
- use of **sliding window**;
- application of **temporal feature engineering**;
- integration between **unsupervised model** and **supervised model**;
- subgroup evaluation;
- critical result analysis;
- ability to explain when a solution helps more and when it helps less.
