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

# Project Overview

Online retail platforms collect large amounts of behavioral data during user sessions. However, most businesses fail to leverage this data effectively to predict customer intent.

The goal of this project is to develop a **classification model capable of predicting whether a user session will result in a purchase.**

The project focuses on:

- Understanding customer browsing behavior  
- Identifying key predictors of purchase intent  
- Building and evaluating machine learning models  
- Translating model outputs into actionable business insights  

---

# Business Problem

E-commerce conversion rates are typically low. In this dataset:

- **~15.5%** of sessions resulted in a purchase  
- **~84.5%** of visitors did not convert  

Without predictive analytics, marketing teams often allocate advertising budgets inefficiently.

A predictive model allows businesses to:

- Identify high-probability buyers
- Personalize marketing strategies
- Reduce advertising waste
- Improve conversion rates

---

# Dataset

The dataset used in this project is the **Online Shoppers Purchasing Intention Dataset**.

Source:  
https://www.kaggle.com/datasets/imakash3011/online-shoppers-purchasing-intention-dataset

The dataset contains **12,330 user sessions and 18 behavioral features** describing browsing activity, session attributes, and visitor characteristics.

![](https://github.com/idayatakinwale/Customer-Purchase-Prediction-in-E-Commerce/blob/main/images/capstone%20dataset.png)

---

# Key Features

### Behavioral Metrics

- Administrative pages visited  
- Informational pages visited  
- Product-related pages visited  
- Bounce rates  
- Exit rates  
- Page values  

### Session Attributes

- Month  
- Special day proximity  
- Weekend indicator  

### Visitor Characteristics

- Visitor type (new vs returning)  
- Browser  
- Operating system  
- Region  
- Traffic source  

### Target Variable

**Revenue (True / False)**  
Indicates whether the session resulted in a purchase.

---

# Project Workflow

The project followed a structured data science pipeline:

1. Business/Data Understanding  
2. Data Exploration (EDA)  
3. Data Preprocessing  
4. Feature Engineering  
5. Model Development  
6. Model Evaluation  
7. Model Optimization  
8. Business Insight Generation  

---

# Data Understanding

The dataset initially contained **12,330 observations** with **18 features** describing online browsing behavior.

Key findings during initial inspection:

- **No missing values** were present.
- **125 duplicate rows** were identified and removed.
- Final dataset size: **12,205 observations**.
- The target variable **Revenue is highly imbalanced**, with only **15.47% positive purchase sessions**.

This imbalance influenced model evaluation and optimization strategies.

---

# Exploratory Data Analysis

EDA revealed several behavioral patterns associated with purchasing decisions.

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

```
12330 → 12205 rows
```

### Outlier Treatment

Extreme values were **capped at the 99th percentile** for:

- Administrative_Duration
- Informational_Duration
- ProductRelated_Duration

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

# Models Implemented

Three classification algorithms were trained and evaluated.

| Model | Accuracy | Precision | Recall | F1 Score | ROC-AUC |
|------|------|------|------|------|------|
| Random Forest | **0.885** | **0.606** | **0.764** | **0.676** | **0.929** |
| Logistic Regression | 0.852 | 0.517 | **0.788** | 0.625 | 0.910 |
| Decision Tree | 0.857 | 0.531 | 0.746 | 0.620 | 0.869 |

---

# Best Model: Random Forest

The **Random Forest model** achieved the strongest overall performance.

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

Confirmed the **stability and robustness** of the Random Forest model across multiple folds.

### Hyperparameter Tuning

`GridSearchCV` was used to tune parameters such as:

- n_estimators
- max_depth
- min_samples_split

This improved predictive performance.

### Threshold Optimization

A custom classification threshold was selected to **maximize the F1-Score**, improving balance between precision and recall.

---

# Marketing Optimization Analysis

### Lift Curve

The lift curve shows that **high-probability predictions significantly outperform random targeting**.

### Cumulative Gains

A key business insight emerged:

> **Targeting the top 20% of predicted customers captures about 65% of all purchases.**

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
