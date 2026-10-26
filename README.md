# Customer Churn Prediction and Retention Optimisation

**Team:** Team ChurnGuard  
**Tagline:** Turning Customer Data into Retention Strategies

## 1. Project Overview

This project explores customer attrition in a banking dataset and prepares the data for a future machine-learning workflow. The aim is to understand which customer characteristics and behaviours are associated with attrition and to carry out the data-cleaning and preprocessing steps needed before model development.

**Current scope:** Exploratory Data Analysis (EDA) and data preprocessing. Model training, model evaluation, churn predictions, and live retention recommendations have not been completed yet.

## 2. Problem Statement

Customer attrition can affect a bank's customer relationships and long-term business performance. Analysing historical customer information can help identify patterns associated with customers who have left. This project examines those patterns and prepares the dataset for possible future churn prediction and retention analysis.

The visualisations show associations in the dataset; they do not prove that a particular feature causes a customer to leave.

## 3. Project Objectives

- Inspect the dataset's structure, data types, and summary statistics.
- Examine the balance between existing and attrited customers.
- Visualise numerical feature distributions and potential outliers.
- Compare numerical features across attrition groups.
- Explore attrition percentages across categorical features.
- Inspect correlations between numerical features.
- Check missing values and duplicate rows.
- Complete justified preprocessing, including outlier treatment, categorical encoding, and feature scaling.

## 4. Dataset

- **Dataset:** Bank Churners
- **File used by the notebook:** `BankChurners.csv`
- **Shape shown in the notebook:** 10,127 rows and 23 columns
- **Target column:** `Attrition_Flag`
- **Target categories:** `Existing Customer` and `Attrited Customer`

The dataset includes customer demographic information, account characteristics, and transaction-related features.

### Important columns to review

- `Attrition_Flag` is the target label for attrition status.
- `CLIENTNUM` is a customer identifier and should not be used as a predictive feature.
- Columns beginning with `Naive_Bayes_Classifier_` should be investigated and excluded from a future prediction workflow if they contain target-derived information or predictions produced using the target. Keeping them may cause target leakage.

## 5. Exploratory Data Analysis Completed So Far

The following graph images are included in this repository under the `images/` folder. They were generated from the uploaded Bank Churners dataset and correspond to the analyses described below. The notebook also contains individual boxplots for numerical features; those are not all duplicated here to keep the README readable.

### 5.1 Dataset inspection

The notebook displays a sample of the data, lists column names, checks data types with `df.info()`, and displays descriptive statistics with `df.describe()`.

### 5.2 Customer attrition distribution

A count plot titled **Customer Attrition Distribution** compares the number of existing and attrited customers. The notebook also prints counts and percentages for each target category.

**Why it matters:** This shows the class distribution and helps identify whether attrited customers are a minority class.

![Customer attrition distribution](images/customer-attrition-distribution.png)

*Figure 1. Number of existing and attrited customers.*

### 5.3 Numerical feature distributions

Histograms are plotted for numerical columns after excluding `CLIENTNUM` and columns beginning with `Naive_Bayes_Classifier_`.

**Why it matters:** Histograms help reveal skewness, concentration, spread, and unusual values that may need further investigation.

![Numerical feature distributions](images/numerical-feature-distributions.png)

*Figure 2. Distributions of numerical features, excluding the customer ID and Naive Bayes output columns.*

### 5.4 Numerical feature boxplots

The notebook creates a boxplot for each selected numerical feature.

**Why it matters:** Boxplots help flag observations outside the IQR-based whiskers. Such observations are potential outliers, not automatically data errors.

### 5.5 Correlation heatmap

The notebook includes code for a **Correlation Heatmap** of numerical features.

**Why it matters:** Correlation can reveal strongly related features and possible redundancy. Correlation does not establish causation.

![Correlation heatmap](images/correlation-heatmap.png)

*Figure 3. Correlations between selected numerical features.*

### 5.6 Attrition rate across categorical features

For categorical columns with no more than 15 unique values, the notebook calculates a row-normalised cross-tabulation and plots the percentage of attrited customers by category. This includes relevant categories such as gender, education, marital status, income category, and card category when present in the data.

**Why it matters:** Percentages make categories with different numbers of customers easier to compare than raw counts alone.

![Attrition rates across categorical features](images/attrition-rate-categorical-features.png)

*Figure 4. Percentage of attrited customers across selected categorical features.*

### 5.7 Numerical features by attrition status

The notebook creates boxplots comparing each selected numerical feature for `Existing Customer` and `Attrited Customer`.

**Why it matters:** These comparisons help show whether feature distributions differ between the two attrition groups. Any differences should be interpreted as associations, not proof of causation.

![Numerical features by attrition status](images/numerical-features-by-attrition.png)

*Figure 5. Selected numerical features compared between existing and attrited customers.*

### 5.8 IQR outlier summary

The notebook calculates the first quartile (Q1), third quartile (Q3), interquartile range (IQR), lower and upper limits, outlier counts, and outlier percentages for selected numerical columns.

The IQR limits are:

- `IQR = Q3 - Q1`
- `Lower limit = Q1 - 1.5 × IQR`
- `Upper limit = Q3 + 1.5 × IQR`

Values outside these limits are flagged for investigation.

## 6. Preprocessing: Work Remaining

The notebook includes basic duplicate and target-missing-value handling, an IQR outlier summary, an outlier-removal demonstration, categorical encoding, and demonstrations of standardisation and Min-Max scaling. The results and assumptions still need to be reviewed and validated before treating the preprocessing as final.

### 6.1 Finalise data cleaning

- Review the missing-value results and decide how to handle missing values, if any.
- Investigate duplicate rows before removing them.
- Confirm the final dataset shape after cleaning.
- Keep the original data available so that cleaning decisions can be compared and reproduced.

### 6.2 Finalise feature selection

- Exclude `CLIENTNUM` from predictive input features.
- Investigate the `Naive_Bayes_Classifier_` columns and remove them from the future feature set if they create target leakage.
- Keep `Attrition_Flag` as the target, not as an input feature.
- Review strongly correlated or redundant features before deciding whether any should be removed.

### 6.3 Decide how to treat outliers

The notebook currently calculates IQR outlier counts and percentages. The next step is to review the results feature by feature and choose a justified treatment.

- **Capping:** Limit extreme values to chosen lower and upper bounds while retaining the row.
- **Trimming:** Remove rows that meet a justified outlier rule, which can discard legitimate customer records.
- **No change:** Keep legitimate extreme values when they represent plausible customer behaviour.

Do not automatically remove every row that contains an IQR outlier. Features such as credit limits and transaction amounts may have legitimate extreme values. Bounded features and discrete counts should also be reviewed before applying a generic rule.

The notebook defines `df_clean` and `outlier_columns` in earlier cells. Before accepting the result, review whether removing a row for an IQR outlier in any feature removes too many legitimate customer records, and compare row counts before and after treatment.

### 6.4 Encode categorical features

- Identify categorical input columns.
- Use one-hot encoding for nominal categories where appropriate.
- Convert `Attrition_Flag` to a binary target only when needed, with the class mapping documented.
- Do not assign arbitrary numeric values to nominal categories if that would imply a false ordering.

### 6.5 Scale numerical features

- Demonstrate `StandardScaler` for standardisation.
- Demonstrate `MinMaxScaler` as an alternative where useful.
- Do not scale the target column or identifier columns.
- If a model is added later, fit scalers on training data only and use the fitted scaler to transform validation/test data.

### 6.6 Validate the final preprocessed data

- Check the final shape and data types.
- Confirm that no unintended missing values remain.
- Confirm that categorical encoding worked as expected.
- Review the scaled feature ranges.
- Verify that the target and any leakage-prone columns are not included in the input features.

## 7. Tools and Libraries

- Python
- Jupyter Notebook
- pandas
- NumPy
- Matplotlib
- Seaborn
- scikit-learn preprocessing tools (planned for scaling and encoding)

## 8. How to Run the Notebook

1. Place the notebook and `BankChurners.csv` in the same folder, or update the CSV path in the notebook.
2. Install the required packages:

   ```bash
   pip install pandas numpy matplotlib seaborn scikit-learn jupyter
   ```

3. Open Jupyter Notebook or JupyterLab.
4. Open `customer_churn_analysis.ipynb`.
5. Run the cells in order and resolve any undefined variables before running the affected cells.

## 9. Current Project Status

| Component | Status |
|---|---|
| Dataset loading and inspection | Started / present in notebook |
| Target distribution plot | Present in notebook |
| Numerical histograms | Present in notebook |
| Numerical boxplots | Present in notebook |
| Correlation heatmap | Present in notebook and included above |
| Categorical attrition-rate plots | Present in notebook |
| Numerical features by attrition | Present in notebook |
| Missing-value and duplicate checks | Present; results need review |
| IQR outlier summary | Present in notebook |
| Outlier treatment demonstration | Present; review whether row removal is justified |
| Categorical encoding | Demonstrated; validate encoded features |
| Standardisation and normalisation | Demonstrated; use train-only fitting when modelling |
| Final preprocessing validation | Still required |
| Model training and evaluation | Not included yet |

## 10. Limitations and Future Work

The current project is focused on EDA and preprocessing. It does not yet train a churn classifier, report predictive performance, produce churn probabilities, or calculate the financial return of retention actions.

After preprocessing is complete, possible future work includes a leakage-safe train/test split, training and comparing classification models, evaluating precision/recall/F1-score and ROC-AUC, and translating validated predictions into a retention-priority workflow. These are future extensions, not current results.

---

**Project title:** Customer Churn Prediction and Retention Optimisation Using Machine Learning  
**Team:** Team ChurnGuard  
