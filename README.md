# Customer-Purchase-Prediction-in-E-Commerce
This is a machine learning project that predicts whether an online shopper will complete a purchase based on browsing behavior and session attributes.  

This project demonstrates how predictive analytics can help e-commerce businesses identify high-intent visitors, improve marketing efficiency, and optimize revenue generation.

![](https://github.com/idayatakinwale/Customer-Purchase-Prediction-in-E-Commerce/blob/main/images/project%20image.png)

---

# Table of Contents

- [Project Workflow](#project-workflow)
- [Understanding the Problem](#understanding-the-problem)
- [Dataset Information](#dataset-information)
- [Data Cleaning and Validation](#data-cleaning-and-validation)
- [Exploratory Data Analysis](#exploratory-data-analysis)
- [Data Preprocessing](#data-preprocessing)
- [Feature Engineering](#feature-engineering)
- [Model Building and Evaluation](#model-building-and-evaluation)
- [Model Optimization](#model-optimization)
- [Marketing Optimization Analysis](#marketing-optimization-analysis)
- [Key Behavioral Insights](#key-behavioral-insights)
- [Business Recommendations](#business-recommendations)
- [Tools and Technologies](#tools-and-technologies)
- [Project Structure](#project-structure)
- [Future Improvements](#future-improvements)
- [Contributors](#contributors)

---

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

The dataset used in this project is the **Online Shoppers Purchasing Intention Dataset**, which was obtained from [Kaggle](https://www.kaggle.com/datasets/imakash3011/online-shoppers-purchasing-intention-dataset)

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


## Categorical Insights

### Visitor Type

| Visitor Type | Purchase Rate |
|------|------|
| New Visitor | 24.9% |
| Other | 18.8% |
| Returning Visitor | 13.9% |

Although returning visitors convert less frequently per session, they represent the **majority of overall traffic**, making them important for total revenue generation.


### Weekend Effect

- Weekend purchase rate: **17.4%**
- Weekday purchase rate: **14.9%**

Sessions occurring during weekends show slightly higher purchase probabilities.


### Seasonal Trends (Month)

Highest conversion months:

- **November (25.35%)**
- **October (20.95%)**

Lowest conversion month:

- **February (1.63%)**

This reflects strong **seasonal shopping behavior**, particularly during major retail events toward the end of the year.


### Traffic Source Effect (TrafficType)

Purchase rates vary significantly across different **TrafficType categories**, indicating that certain traffic acquisition channels are more effective at driving conversions.

Examples include:

- **TrafficType 16 → 33.33% purchase rate**
- **TrafficType 7 → 30.00% purchase rate**

This suggests that marketing channels associated with these traffic types are generating **higher-intent visitors**.

#### Purchase Rate by Traffic Type

![](https://github.com/idayatakinwale/Customer-Purchase-Prediction-in-E-Commerce/blob/main/images/purchase%20rate%20by%20traffic%20type.png)

## Correlation Analysis

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

Extreme values were **capped at the 99th percentile** for columns:

- Administrative_Duration
- Informational_Duration
- ProductRelated_Duration

![](https://github.com/idayatakinwale/Customer-Purchase-Prediction-in-E-Commerce/blob/main/images/outlier%20treatment.png)

This method helps to mitigate the impact of extreme values on model performance without discarding valuable data, particularly relevant for features with skewed distributions.

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

**Overall Accuracy**: With an accuracy of 0.8853, Random Forest also outperforms the other models in correctly classifying both purchasing and non-purchasing sessions.

**Balanced Precision & Recall**

| Metric | Value |
|------|------|
| Precision | 0.61 |
| Recall | 0.76 |

This balance is ideal for marketing use cases where both **identifying buyers** and **limiting false positives** matter.

In summary, the **Random Forest** model demonstrates robust performance across key metrics, making it the most suitable choice for predicting customer purchases in this scenario due to its strong ability to handle class imbalance and its superior discriminative capabilities.

---

# Model Optimization

Several optimization techniques were applied.

### Cross Validation

This confirmed the **stability and robustness** of the Random Forest model across multiple folds, indicating that its performance is reliable and not overly sensitive to the specific train-test split.

![](https://github.com/idayatakinwale/Customer-Purchase-Prediction-in-E-Commerce/blob/main/images/cross%20validation%20code.png)

![](https://github.com/idayatakinwale/Customer-Purchase-Prediction-in-E-Commerce/blob/main/images/cross%20validation%20result.png)

### Hyperparameter Tuning

`GridSearchCV` was used to tune parameters such as:

- n_estimators
- max_depth
- min_samples_split

This improved predictive performance, (specifically maximizing ROC-AUC) over the baseline Random Forest configuration.


### Threshold Optimization

A custom classification threshold was selected to **maximize the F1-Score**, improving balance between precision and recall, especially in imbalanced datasets like this where identifying the positive class (purchasers) is important. This optimization ensures that the model's predictions are aligned with a desired balance between false positives and false negatives.

![](https://github.com/idayatakinwale/Customer-Purchase-Prediction-in-E-Commerce/blob/main/images/optimized%20random%20forest.png)

---

## Marketing Optimization Analysis

### Lift Curve

![](https://github.com/idayatakinwale/Customer-Purchase-Prediction-in-E-Commerce/blob/main/images/lift%20curve.png)

The lift curve shows that **high-probability predictions significantly outperform random targeting**. For instance, the top deciles (groups of customers with the highest predicted probabilities) show a much higher conversion rate than the average conversion rate across all customers.

### Cumulative Gains

A key business insight emerged:

![](https://github.com/idayatakinwale/Customer-Purchase-Prediction-in-E-Commerce/blob/main/images/gains%20curve.png)

> **Targeting the top 20% of predicted customers captures about 65% of total conversions, representing a significant improvement over random targeting (which would capture only 20%). That is marketing optimization..**
This dramatically improves marketing efficiency compared with random campaigns.

---

# Key Insights

### 1. Low Purchase Conversion Rate
Only **15.47% of user sessions resulted in a purchase**, indicating that the majority of visitors browse without completing a transaction. This highlights the importance of identifying and targeting users with higher purchase intent.

### 2. Page Value Strongly Predicts Purchase Intent
The **PageValues** feature showed the strongest relationship with purchases. Users who completed a purchase had significantly higher PageValues compared to non-purchasers, making it a key indicator of buying intent.

### 3. Higher Engagement Increases Purchase Probability
Purchasing sessions were associated with **longer browsing durations**, particularly on **product-related pages**. This suggests that users who spend more time exploring products are more likely to convert.

### 4. Bounce and Exit Rates Indicate Low Conversion Potential
Sessions with **high BounceRates and ExitRates** were less likely to result in purchases. Visitors who leave quickly or exit frequently tend to have a much lower likelihood of converting.

### 5. Seasonal Trends Influence Customer Purchases
Conversion rates varied across months, with **October and November recording the highest purchase rates**, likely due to seasonal promotions and increased shopping activity during that period.

### 6. Random Forest Achieved the Best Model Performance
Among the models tested, the **Random Forest model delivered the best performance**, achieving a **ROC-AUC score of 0.929**, indicating strong ability to distinguish between purchasing and non-purchasing sessions.

---

# Business Recommendations

### 1. Implement Predictive Customer Targeting
Use the predictive model to identify high-probability buyers and prioritize marketing efforts toward them. This can include:
- Personalized product offers
- Exit-intent promotions
- Retargeting campaigns

Focusing on users with higher predicted purchase probability can significantly improve marketing efficiency and ROI.

---

### 2. Improve Website Engagement and User Experience
Since purchasing sessions showed **higher engagement and lower bounce rates**, improving the website experience is essential. Businesses should focus on:

- Faster page load speeds
- Better website navigation
- High-quality product pages
- Mobile optimization

Reducing bounce and exit rates can increase the likelihood of users progressing toward a purchase.

---

### 3. Optimize High-Value Pages
The **PageValues** feature was identified as the strongest predictor of purchase intent. Businesses should identify pages that contribute most to conversion and optimize them through:

- Better content and product information
- Clear calls-to-action
- Strategic placement within the customer journey

---

### 4. Leverage Seasonal Marketing Opportunities
The analysis showed higher purchase rates during **October and November**, indicating strong seasonal demand.

Businesses should increase marketing investments and promotional campaigns during peak shopping periods such as:
- October
- November
- December

This can help maximize conversions during high-demand periods.

---

### 5. Use Visitor Segmentation for Personalized Marketing
Different visitor groups exhibit different purchasing behaviors.

- **New visitors** can be encouraged with onboarding offers or first-time discounts.
- **Returning visitors** can be re-engaged with personalized recommendations and targeted promotions.

Segmenting customers allows businesses to deliver more relevant marketing experiences.

---

### 6. Continuously Improve the Predictive Model
To maintain model effectiveness over time:

- Retrain models with updated behavioral data
- Monitor model performance regularly
- Experiment with advanced algorithms such as **XGBoost** or **LightGBM**

Continuous improvement ensures the model adapts to changing customer behavior.

---

# Conclusion

This project demonstrated how behavioral data from e-commerce sessions can be leveraged to predict customer purchase intent. Through exploratory data analysis, key behavioral patterns influencing conversions were identified, including the importance of page value, user engagement duration, and bounce/exit behavior.

Several machine learning models were evaluated, with the **Random Forest model achieving the best performance**, demonstrating strong ability to distinguish between purchasing and non-purchasing sessions. The results show that predictive modeling can effectively identify high-probability buyers and provide valuable insights for marketing optimization.

By applying these insights, businesses can improve customer targeting, enhance website engagement, and allocate marketing resources more efficiently. Overall, the project highlights the potential of data-driven strategies to increase conversion rates and support better decision-making in online retail environments.

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

[GitHub](https://github.com/samsology/) 

[LinkedIn](https://www.linkedin.com/in/samuel-johnson-766b2a337/)

**Idayat Akinwale**

[GitHub](https://github.com/idayatakinwale)

[LinkedIn](https://www.linkedin.com/in/idayat-akinwale)

**Oliver  Ocran**

[GitHub](https://github.com/O-HANSON)

[LinkedIn](https://www.linkedin.com/in/oliverocran)

**Akinsiku Oluwadamilola**

[GitHub](https://github.com/oluwadamilolaakinsiku)

[LinkedIn](https://www.linkedin.com/in/akinsiku-oluwadamilola)

**Daniel Ofosu Ampadu**

[GitHub](https://github.com/dannieoa)

[LinkedIn](https://www.linkedin.com/in/ampadudanielofosu)

**Omotola Olufunmilayo**

[GitHub](https://github.com/olufunmi-bit)

[LinkedIn](https://www.linkedin.com/in/olufunmi-omotola)

**Kolo Stephen**

[GitHub](https://github.com/kolosteve)

[LinkedIn](http://www.linkedin.com/in/kolo-stephen-641a792a3)

**Adekunle Mahmud**

[GitHub](https://github.com/adekunlemahmud)

[LinkedIn](https://www.linkedin.com/in/mahmud-adekunle-0649a7198/)


**Team Dynamo**

Project collaboration and analytics support  
Capstone project development
