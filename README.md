# Customer-Purchase-Prediction-in-E-Commerce
This is a machine learning project that predicts whether an online shopper will complete a purchase based on browsing behavior and session attributes.  

This project demonstrates how predictive analytics can help e-commerce businesses identify high-intent visitors, improve marketing efficiency, and optimize revenue generation.

---

# Table of Contents

- [Project Overview](#project-overview)
- [Business Problem](#business-problem)
- [Dataset](#dataset)
- [Key Features](#key-features)
- [Project Workflow](#project-workflow)
- [Data Understanding](#data-understanding)
- [Exploratory Data Analysis](#exploratory-data-analysis)
- [Correlation Analysis](#correlation-analysis)
- [Data Preprocessing](#data-preprocessing)
- [Feature Engineering](#feature-engineering)
- [Models Implemented](#models-implemented)
- [Best Model: Random Forest](#best-model-random-forest)
- [Model Optimization](#model-optimization)
- [Marketing Optimization Analysis](#marketing-optimization-analysis)
- [Key Behavioral Insights](#key-behavioral-insights)
- [Business Recommendations](#business-recommendations)
- [Tools and Technologies](#tools-and-technologies)
- [Project Structure](#project-structure)
- [Future Improvements](#future-improvements)
- [Contributors](#contributors)

# Project Workflow

The key steps followed to achieve the objective of this project are:

1. **Understanding the Problem**
2. **Dataset Information**
3. **Data Cleaning and Validation**
4. **Exploratory Data Analysis (EDA)**
5. **Data Preprocessing**
6. **Feature Engineering**
7. **Model Development**
8. **Model Evaluation**
9. **Model Optimization**
10. **Business Insight Generation**

---

# Understanding the Problem

Online retail platforms generate large volumes of behavioral data as users browse through products, pages, and categories during their sessions. Despite the availability of this data, many businesses do not effectively leverage it to understand customer intent or predict purchasing behavior.

One of the major challenges in e-commerce is the **low conversion rate**, where only a small fraction of visitors actually complete a purchase. In this dataset, approximately **15.5% of user sessions resulted in a transaction**, while **about 84.5% of sessions ended without a purchase**.

Without the ability to identify which visitors are most likely to convert, marketing teams often distribute advertising budgets and promotional efforts inefficiently. This results in missed opportunities to engage high-intent customers and maximize revenue.

The objective of this project is therefore to build a **machine learning classification model** that can predict whether a user session will result in a purchase based on browsing behavior and session attributes.

By leveraging predictive analytics, businesses can:

- Identify **high-probability buyers**
- Personalize **marketing and promotional strategies**
- Reduce **advertising waste**
- Improve **conversion rates and revenue generation**

This project explores how user behavior metrics such as **page engagement, browsing duration, bounce rates, and traffic sources** can be used to predict purchasing intent and support more data-driven decision-making in e-commerce environments.

---

# Dataset Information

The dataset used in this project is the **Online Shoppers Purchasing Intention Dataset**, which was obtained from [Kaggle Dataset](https://www.kaggle.com/datasets/imakash3011/online-shoppers-purchasing-intention-dataset)

The dataset contains **12,330 user sessions (Rows) and 18 behavioral features (Columns)** describing browsing activity, session attributes, and visitor characteristics.

It was then loaded into the workbook

![](https://github.com/idayatakinwale/Customer-Purchase-Prediction-in-E-Commerce/blob/main/images/loaded%20dataset.png)

**Here is the Data Dictionary**;

The dataset contains **18 features** describing user browsing behavior during an e-commerce session.  
These include **10 numerical variables** and **8 categorical variables**, with **Revenue** serving as the target variable.

| Feature | Type | Description |
|--------|------|-------------|
| Administrative | Numerical | Number of administrative pages visited during the session. |
| Administrative_Duration | Numerical | Total time spent on administrative pages during the session. |
| Informational | Numerical | Number of informational pages visited. |
| Informational_Duration | Numerical | Total time spent on informational pages. |
| ProductRelated | Numerical | Number of product-related pages visited. |
| ProductRelated_Duration | Numerical | Total time spent on product-related pages. |
| BounceRates | Numerical | Percentage of visitors who leave the site after viewing only one page. |
| ExitRates | Numerical | Percentage of exits from a page relative to total page views. |
| PageValues | Numerical | Average value of a page based on visits before completing a transaction. |
| SpecialDay | Numerical | Indicates closeness of the visit to special shopping days (e.g., Valentine’s Day). |
| Month | Categorical | Month when the session occurred. |
| OperatingSystems | Categorical | Operating system used by the visitor. |
| Browser | Categorical | Web browser used during the session. |
| Region | Categorical | Geographic region of the visitor. |
| TrafficType | Categorical | Source of the website traffic. |
| VisitorType | Categorical | Type of visitor (Returning, New, or Other). |
| Weekend | Categorical | Indicates whether the session occurred on a weekend. |
| Revenue | Target | Indicates whether the session resulted in a purchase (True/False). |
---

# Data Cleaning and Validation

Before performing any analysis, the dataset was inspected to verify its quality and ensure that it was suitable for modeling. The validation process focused on checking for missing values, identifying duplicate records, and assessing the distribution of the target variable.

**Checking for Missing Values**

The dataset was first examined to determine whether any columns contained missing values.


![](https://github.com/idayatakinwale/Customer-Purchase-Prediction-in-E-Commerce/blob/main/images/null%20values.png)

The results show that **no missing values were present in any of the columns**, indicating that the dataset was already complete and did not require imputation.

**Checking for Duplicate Records**

Next, the dataset was checked for duplicate observations that could introduce bias into the analysis.


![](https://github.com/idayatakinwale/Customer-Purchase-Prediction-in-E-Commerce/blob/main/images/duplicates.png)

The inspection revealed **125 duplicate rows** in the dataset. These duplicates were removed to ensure that each observation represented a unique user session.

### Dataset Status After Cleaning

After removing duplicate records, the dataset contained:

- **12,205 total observations**
- **18 features describing user behavior and session attributes**

Additionally, an inspection of the target variable revealed a **significant class imbalance**:

- **15.47%** of sessions resulted in a purchase  
- **84.53%** of sessions did not result in a purchase

**This imbalance is an important consideration when building predictive models, as it influences model evaluation strategies and performance metrics**.

---

# Exploratory Data Analysis

EDA revealed several behavioral patterns associated with purchasing decisions.

First, the required libraries were imported;
![](https://github.com/idayatakinwale/Customer-Purchase-Prediction-in-E-Commerce/blob/main/images/LIBRARIES%20USED.png)

## Univariate Insights

- The **Revenue variable is strongly imbalanced**  
  - 84.53% → No purchase  
  - 15.47% → Purchase

![](https://github.com/idayatakinwale/Customer-Purchase-Prediction-in-E-Commerce/blob/main/images/purchase%20distribution.png)


- Most numerical features are **right-skewed**, with many sessions having very low activity.

- Several features contained outliers, especially:
  - PageValues
  - Informational
  - Informational_Duration
  - BounceRates

Example:

| Feature | Outliers |
|------|------|
| PageValues | 22.14% |
| Informational | 21.34% |

---

## Bivariate Insights

### Behavioral Metrics vs Purchase

**PageValues**

- Strongest indicator of purchase intent
- Purchasers average: **27.27**
- Non-purchasers average: **1.98**
- **1279% relative difference**

**BounceRates & ExitRates**

Purchasers exhibit significantly lower values:

| Metric | Purchasers | Non-Purchasers |
|------|------|------|
| BounceRates | 0.005 | 0.025 |
| ExitRates | 0.020 | 0.047 |

Lower bounce and exit rates strongly correlate with higher purchase probability.

**User Engagement**

Purchasing sessions spend significantly more time on:

- Product pages
- Informational pages
- Administrative pages

---

## Categorical Insights

### Visitor Type

| Visitor Type | Purchase Rate |
|------|------|
| New Visitor | 24.9% |
| Other | 18.8% |
| Returning Visitor | 13.9% |

Although returning visitors convert less frequently per session, they represent the **majority of overall traffic**, making them important for total revenue generation.

---

### Weekend Effect

- Weekend purchase rate: **17.4%**
- Weekday purchase rate: **14.9%**

Sessions occurring during weekends show slightly higher purchase probabilities.

---

### Seasonal Trends (Month)

Highest conversion months:

- **November (25.35%)**
- **October (20.95%)**

Lowest conversion month:

- **February (1.63%)**

This reflects strong **seasonal shopping behavior**, particularly during major retail events toward the end of the year.

---

### Traffic Source Effect (TrafficType)

Purchase rates vary significantly across different **TrafficType categories**, indicating that certain traffic acquisition channels are more effective at driving conversions.

Examples include:

- **TrafficType 16 → 33.33% purchase rate**
- **TrafficType 7 → 30.00% purchase rate**

This suggests that marketing channels associated with these traffic types are generating **higher-intent visitors**.

#### Purchase Rate by Traffic Type

![](https://github.com/idayatakinwale/Customer-Purchase-Prediction-in-E-Commerce/blob/main/images/purchase%20rate%20by%20traffic%20type.png)

---

# Correlation Analysis

![](https://github.com/idayatakinwale/Customer-Purchase-Prediction-in-E-Commerce/blob/main/images/correlation%20matrix.png)

The strongest predictors of purchase behavior include:

| Feature | Correlation with Revenue |
|------|------|
| PageValues | 0.493 |
| ExitRates | -0.207 |
| ProductRelated | 0.159 |
| ProductRelated_Duration | 0.152 |
| BounceRates | -0.151 |
| Administrative | 0.139 |

**PageValues** clearly stands out as the **most powerful predictor of purchase intent**.

---

# Data Preprocessing

Several preprocessing steps were implemented to improve model performance.

### Duplicate Removal

125 duplicate records were removed:

![](https://github.com/idayatakinwale/Customer-Purchase-Prediction-in-E-Commerce/blob/main/images/drop%20duplicate.png)

```
12330 → 12205 rows
```

### Outlier Treatment

Extreme values were **capped at the 99th percentile** for:

- Administrative_Duration
- Informational_Duration
- ProductRelated_Duration

![](https://github.com/idayatakinwale/Customer-Purchase-Prediction-in-E-Commerce/blob/main/images/outlier%20treatment.png)

This preserved valuable data while reducing the impact of extreme observations.

---

# Feature Engineering

Three additional features were engineered to capture user engagement.

**TotalDuration**

Total time spent across all page categories.

```
Administrative_Duration
+ Informational_Duration
+ ProductRelated_Duration
```

**EngagementRatio**

Measures how much of the browsing time is focused on product pages.

```
ProductRelated_Duration / TotalDuration
```

**BounceToExitRatio**

Captures the relationship between bounce and exit behavior.

```
BounceRates / ExitRates
```

These engineered features help better represent **user engagement behavior**.

---

# Model Building and Evaluation

Three classification algorithms were trained and evaluated.

### Logistic Regression

![](https://github.com/idayatakinwale/Customer-Purchase-Prediction-in-E-Commerce/blob/main/images/LR%20CODE.png)

![](https://github.com/idayatakinwale/Customer-Purchase-Prediction-in-E-Commerce/blob/main/images/LR%20PLOT.png)

### Decision Tree

![](https://github.com/idayatakinwale/Customer-Purchase-Prediction-in-E-Commerce/blob/main/images/decision%20TREE%20CODE.png)

![](https://github.com/idayatakinwale/Customer-Purchase-Prediction-in-E-Commerce/blob/main/images/decision%20TREE%20PLOT.png)

### Random Forest

![](https://github.com/idayatakinwale/Customer-Purchase-Prediction-in-E-Commerce/blob/main/images/RANDOM%20FOREST%20CODE.png)

![](https://github.com/idayatakinwale/Customer-Purchase-Prediction-in-E-Commerce/blob/main/images/RANDOM%20FOREST%20PLOT.png)

### Model Performance Summary

| Model | Accuracy | Precision | Recall | F1 Score | ROC-AUC |
|------|------|------|------|------|------|
| Random Forest | **0.885** | **0.606** | **0.764** | **0.676** | **0.929** |
| Logistic Regression | 0.852 | 0.517 | **0.788** | 0.625 | 0.910 |
| Decision Tree | 0.857 | 0.531 | 0.746 | 0.620 | 0.869 |

![](https://github.com/idayatakinwale/Customer-Purchase-Prediction-in-E-Commerce/blob/main/images/Model%20comparison.png)

The model comparison results show that **Random Forest** performed best with an ROC-AUC of 0.93, meaning it can distinguish between purchasers and non-purchasers with 93% accuracy.

### Key Strengths

**Highest ROC-AUC (0.93)**  
Indicates strong ability to distinguish buyers from non-buyers.

**Highest F1 Score (0.68)**  
Balances precision and recall effectively.

**Balanced Precision & Recall**

| Metric | Value |
|------|------|
| Precision | 0.61 |
| Recall | 0.76 |

This balance is ideal for marketing use cases where both **identifying buyers** and **limiting false positives** matter.

---

# Model Optimization

Several optimization techniques were applied.

### Cross Validation

This confirmed the **stability and robustness** of the Random Forest model across multiple folds.

![](https://github.com/idayatakinwale/Customer-Purchase-Prediction-in-E-Commerce/blob/main/images/cross%20validation%20code.png)

![](https://github.com/idayatakinwale/Customer-Purchase-Prediction-in-E-Commerce/blob/main/images/cross%20validation%20result.png)

### Hyperparameter Tuning

`GridSearchCV` was used to tune parameters such as:

- n_estimators
- max_depth
- min_samples_split

This improved predictive performance.

### Threshold Optimization

A custom classification threshold was selected to **maximize the F1-Score**, improving balance between precision and recall.

![](https://github.com/idayatakinwale/Customer-Purchase-Prediction-in-E-Commerce/blob/main/images/optimized%20random%20forest.png)

---

## Marketing Optimization Analysis

### Lift Curve

The lift curve shows that **high-probability predictions significantly outperform random targeting**.

![](https://github.com/idayatakinwale/Customer-Purchase-Prediction-in-E-Commerce/blob/main/images/lift%20curve.png)

### Cumulative Gains

A key business insight emerged:

![](https://github.com/idayatakinwale/Customer-Purchase-Prediction-in-E-Commerce/blob/main/images/gains%20curve.png)

> **Targeting the top 20% of predicted customers captures about 65% of total conversions, representing a significant improvement over random targeting (which would capture only 20%). That is marketing optimization..**

This dramatically improves marketing efficiency compared with random campaigns.

---

# Key Behavioral Insights

Important predictors of purchasing behavior include:

**High PageValues**

Strongest signal of purchase intent.

**Low BounceRates**

Engaged users are more likely to convert.

**Low ExitRates**

Successful purchase journeys involve deeper browsing.

**Higher Engagement Time**

Purchasers spend significantly more time on product pages.

**Seasonality**

Conversion peaks during major shopping months such as **October and November**.

---

# Business Recommendations

### Strategic Customer Targeting

Use predictive scoring to identify high-probability buyers and prioritize:

- Personalized offers
- Exit-intent promotions
- Retargeting campaigns

### Improve Website Engagement

Reduce bounce and exit rates by improving:

- Page load speed
- Website usability
- Product page quality
- Mobile optimization

### Seasonal Marketing Strategy

Increase marketing investments during peak shopping months such as:

- October
- November
- December

### Visitor Segmentation

- Nurture **new visitors** with targeted onboarding offers.
- Re-engage **returning visitors** with personalized recommendations.

### Continuous Model Improvement

- Retrain models with new behavioral data.
- Experiment with advanced models such as **XGBoost or LightGBM**.

---

# Tools and Technologies

### Programming Language

Python

### Libraries

- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn

### Environment

Jupyter Notebook

---

# Project Structure

```
customer-purchase-prediction/

│
├── data/
│   └── online_shoppers_intention.csv
│
├── notebooks/
│   └── customer_purchase_prediction.ipynb
│
├── README.md
│
└── requirements.txt
```

---

# Future Improvements

Possible extensions include:

- Gradient Boosting models (XGBoost, LightGBM)
- SHAP explainability analysis
- Cost-sensitive learning for imbalanced data
- Real-time prediction deployment via API
- A/B testing integration with marketing systems

---

# Contributors

**Samuel Johnson**  
Data Analyst | Data Science Practitioner

GitHub: *(add link)*  
LinkedIn: *(add link)*  

**Team Dynamo**

Project collaboration and analytics support  
Capstone project development
