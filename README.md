
<div align="center">

# 🛍️ Predicting Online Purchase Intention

### A Machine Learning Approach to E-Commerce Conversion Prediction

<p>
  <img src="https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python"/>
  <img src="https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white" alt="Pandas"/>
  <img src="https://img.shields.io/badge/Scikit--learn-F7931E?style=for-the-badge&logo=scikitlearn&logoColor=white" alt="Scikit-learn"/>
  <img src="https://img.shields.io/badge/Machine%20Learning-FF79C6?style=for-the-badge" alt="Machine Learning"/>
</p>

**Can we predict whether an online shopping session will result in a purchase?**

An academic Data Mining project exploring customer behavior, supervised classification, and data-driven decision-making in e-commerce.

</div>

---

## 📑 Table of Contents

- [1. Project Overview](#-1-project-overview)
- [2. Dataset Description](#-2-dataset-description)
- [3. Problem Statement](#-3-problem-statement)
- [4. Data Preparation](#-4-data-preparation)
- [5. Methodology](#-5-methodology)
- [6. Results and Model Comparison](#-6-results-and-model-comparison)
- [7. Model Interpretation](#-7-model-interpretation)
- [8. Business Insights](#-8-business-insights)
- [9. Limitations and Future Improvements](#-9-limitations-and-future-improvements)
- [10. Technologies Used](#-10-technologies-used)

---

## 🎯 1. Project Overview

In e-commerce, understanding why some visitors complete a purchase while others leave without buying is essential for improving conversion rates and optimizing marketing strategies.

This project applies **supervised machine learning classification techniques** to predict whether an online shopping session will result in a purchase.

Using behavioral, session-level, and technical data, three classification models are trained and evaluated to identify the most effective approach and uncover the factors most strongly associated with purchase intention.

### Objectives

- Predict whether a browsing session will result in a purchase.
- Compare the performance of three classification algorithms.
- Identify the most influential features associated with purchasing behavior.
- Translate model results into actionable business recommendations.

---

## 📊 2. Dataset Description

The project uses the **Online Shoppers Purchasing Intention Dataset**, which contains session-level information from an e-commerce website.

Each observation represents a browsing session, including page interactions, time spent on different pages, session quality indicators, and technical or temporal characteristics.

**Dataset source:** [UCI Machine Learning Repository — Online Shoppers Purchasing Intention Dataset](https://archive.ics.uci.edu/dataset/468/online+shoppers+purchasing+intention+dataset)

### Dataset at a Glance

| Attribute | Description |
|---|---|
| Original observations | 12,330 |
| Observations after duplicate removal | 12,205 |
| Original variables | 18 |
| Features after categorical encoding | 26 |
| Target variable | `Revenue` |
| Task | Binary classification |
| Training set | 80% — 9,764 observations |
| Testing set | 20% — 2,441 observations |

### Feature Categories

| Category | Examples |
|---|---|
| Page interaction | `Administrative`, `ProductRelated`, `Informational` |
| Session quality | `BounceRates`, `ExitRates`, `PageValues` |
| Session context | `Month`, `VisitorType`, `Weekend` |
| Technical context | `OperatingSystems`, `Browser`, `Region`, `TrafficType` |
| Target | `Revenue` |

The target variable `Revenue` indicates whether a session resulted in a purchase:

- `0` — No purchase
- `1` — Purchase

### Target Variable Distribution

The dataset is imbalanced: approximately 84.5% of sessions did not result in a purchase, while 15.5% did.

**Figure 1. Distribution of the target variable Revenue across all sessions**

<!-- Add Figure 1 here -->
<p align="center">
  <img src="images/figure1-target-distribution.png" alt="Figure 1: Distribution of the target variable Revenue across all sessions" width="80%"/>
</p>

---

## ❓ 3. Problem Statement

The central problem is to determine whether behavioral and session-level information can help predict a visitor's likelihood of completing a purchase.

Accurate predictions could support e-commerce businesses in:

- **Conversion optimization:** identifying high-intent sessions and improving the purchasing journey.
- **Marketing allocation:** directing campaigns toward more promising audiences.
- **Personalization:** tailoring recommendations and offers to user behavior.
- **Website optimization:** identifying behavioral signals associated with successful conversions.
- **Operational planning:** supporting better resource allocation across customer support and marketing activities.

### Research Questions

1. Can session-level data predict whether a visitor will complete a purchase?
2. Which features provide the strongest predictive signals?
3. Which classification model offers the best balance between precision and recall?

---

## 🧹 4. Data Preparation

Before model training, the dataset was processed to improve data quality and ensure a reliable evaluation workflow.

### Preprocessing Steps

**1. Missing-value analysis**

All 18 original columns were checked for missing values. No missing values were found, so imputation was unnecessary.

**2. Duplicate removal**

A total of 125 duplicate records were removed, reducing the dataset from 12,330 to 12,205 observations.

**3. Categorical encoding**

The `Month` and `VisitorType` variables were encoded using one-hot encoding with `drop_first=True`. The Boolean variables `Weekend` and `Revenue` were converted to integer values.

**4. Stratified train/test split**

The data was divided into 80% training data and 20% testing data using `random_state=42` and stratification to preserve the target-class proportions.

**5. Feature scaling**

`StandardScaler` was fitted on the training data and applied to the test data for Logistic Regression. Decision Tree and Random Forest used unscaled features.

This approach helped prevent data leakage during scaling and ensured that preprocessing was appropriate for each algorithm.

---

## 🤖 5. Methodology

Three supervised classification algorithms were implemented and compared.

### 5.1 Logistic Regression

Logistic Regression served as the baseline model for binary classification.

- Provides a relatively simple benchmark.
- Models the relationship between input features and purchase probability.
- Uses scaled features because it is sensitive to differences in feature magnitude.
- Configured with `max_iter=1000`.

### 5.2 Decision Tree

Decision Tree was selected to capture non-linear relationships between browsing behavior and purchasing outcomes.

- Learns threshold-based decision rules.
- Captures interactions between behavioral variables.
- Offers an interpretable alternative to ensemble methods.
- Configured with `max_depth=10` to limit tree complexity.

### 5.3 Random Forest

Random Forest combines predictions from multiple decision trees to improve predictive stability and reduce the risk of overfitting compared with a single tree.

- Captures complex, non-linear patterns.
- Combines predictions from 200 trees.
- Provides feature importance scores for model interpretation.
- Configured with `n_estimators=200` and `max_depth=10`.

### Evaluation Metrics

The models were evaluated using four metrics:

- **Accuracy:** proportion of all predictions that are correct.
- **Precision:** proportion of predicted purchases that were actual purchases.
- **Recall:** proportion of actual purchases correctly identified.
- **F1-score:** harmonic mean of precision and recall.

Because the dataset is imbalanced, F1-score is particularly useful for comparing the models' ability to identify purchasing sessions without relying on accuracy alone.

---

## 📈 6. Results and Model Comparison

The following results were obtained on the held-out test set.

| Model | Accuracy | Precision | Recall | F1-score |
|---|---:|---:|---:|---:|
| Logistic Regression | 88.94% | 77.18% | 41.62% | 0.541 |
| Decision Tree | 88.90% | 65.46% | 61.52% | 0.634 |
| **Random Forest** | **90.25%** | 74.83% | 56.81% | **0.646** |

*Note: Values are reported from the project analysis.*

### Key Findings

- **Random Forest achieved the highest accuracy and F1-score**, making it the strongest overall model according to these evaluation metrics.
- Logistic Regression achieved the highest precision but had the lowest recall, missing many actual purchasing sessions.
- Decision Tree achieved the highest recall, identifying more purchasing sessions than the other two models, but with lower precision.
- Random Forest offered a stronger overall balance between precision and recall than Logistic Regression, while achieving the best F1-score.

### F1-score Comparison

**Figure 3. F1-score comparison across Logistic Regression, Decision Tree, and Random Forest**

<!-- Add Figure 3 here -->
<p align="center">
  <img src="images/figure3-f1-score-comparison.png" alt="Figure 3: F1-score comparison across the three classification models" width="85%"/>
</p>

---

## 🔍 7. Model Interpretation

### 7.1 Confusion Matrix — Random Forest

The Random Forest confusion matrix provides a more detailed view of the model's predictions on the 2,441 test observations.

**Figure 2. Confusion matrix for Random Forest on the test set**

<!-- Add Figure 2 here -->
<p align="center">
  <img src="images/figure2-random-forest-confusion-matrix.png" alt="Figure 2: Confusion matrix for Random Forest on the test set" width="75%"/>
</p>

| Actual / Predicted | No Purchase | Purchase |
|---|---:|---:|
| No Purchase | 1,986 | 73 |
| Purchase | 165 | 217 |

The model correctly classified 1,986 non-purchase sessions and 217 purchase sessions.

It produced 73 false positives and 165 false negatives. These results illustrate an important trade-off: the model is relatively precise when predicting a purchase, but it still misses a considerable number of actual buyers.

### 7.2 Feature Importance — Random Forest

Feature importance analysis was used to identify the variables contributing most to the Random Forest model's predictions.

**Figure 4. Top 10 most important features ranked by Random Forest importance score**

<!-- Add Figure 4 here -->
<p align="center">
  <img src="images/figure4-random-forest-feature-importance.png" alt="Figure 4: Top 10 features ranked by Random Forest importance" width="85%"/>
</p>

### Top 10 Features

| Rank | Feature | Importance |
|---:|---|---:|
| 1 | `PageValues` | 0.506 |
| 2 | `ExitRates` | 0.072 |
| 3 | `ProductRelated_Duration` | 0.072 |
| 4 | `ProductRelated` | 0.055 |
| 5 | `BounceRates` | 0.049 |
| 6 | `Administrative_Duration` | 0.042 |
| 7 | `Administrative` | 0.034 |
| 8 | `Month_Nov` | 0.030 |
| 9 | `Informational_Duration` | 0.021 |
| 10 | `TrafficType` | 0.019 |

`PageValues` was by far the most influential feature, with an importance score of 0.506. It was followed by `ExitRates` and `ProductRelated_Duration`.

These scores reflect the Random Forest model's impurity-based feature importance. They indicate predictive contribution, not causation, and should not be interpreted as probabilities or proof that changing a feature will cause more purchases.

---

## 💡 8. Business Insights

The model comparison and feature importance analysis suggest several potential opportunities for e-commerce businesses.

**1. Prioritize high-value page interactions**

`PageValues` was the strongest predictor in the Random Forest model. Understanding which page interactions are associated with purchases may help businesses optimize high-value pages and the customer journey.

**2. Investigate exit behavior**

`ExitRates` and `BounceRates` ranked among the most important features. Analyzing where visitors leave the website could reveal opportunities to improve navigation, content, and checkout usability.

**3. Understand product engagement**

`ProductRelated` and `ProductRelated_Duration` indicate that product-page browsing behavior carries useful predictive information. Businesses could explore personalized recommendations for engaged visitors.

**4. Prepare for seasonal demand**

`Month_Nov` appeared among the top ten features. This may support further investigation into seasonal conversion patterns and promotional periods.

**5. Balance conversion opportunities against false alarms**

The model's precision and recall should guide how predictions are used. For example, a business intervention with a high cost may prioritize precision, whereas a campaign designed to reach more potential buyers may prioritize recall.

These are potential applications of the model's findings, not guaranteed business outcomes.

---

## ⚠️ 9. Limitations and Future Improvements

Although Random Forest performed best overall in this comparison, several limitations remain.

### Limitations

- **Class imbalance:** only around 15.5% of sessions resulted in a purchase, making buyer identification more challenging.
- **False negatives:** Random Forest missed 165 actual purchase sessions in the test set.
- **Limited hyperparameter optimization:** the models were trained with selected parameter settings rather than a systematic search.
- **Single train/test split:** performance was assessed on one split, so the results may vary across different samples.
- **Feature importance limitations:** impurity-based importance does not establish causal relationships and can favor certain feature types.

### Future Improvements

- Test class weighting, oversampling with SMOTE, or undersampling to address class imbalance.
- Use `GridSearchCV` or `RandomizedSearchCV` to optimize model parameters.
- Apply cross-validation for more robust performance estimates.
- Explore precision-recall curves and threshold tuning.
- Investigate additional metrics and the business costs of false positives and false negatives.

---

## 🧰 10. Technologies Used

| Technology | Purpose |
|---|---|
| Python | Data analysis and model development |
| Pandas | Data manipulation and preprocessing |
| Scikit-learn | Classification, feature scaling, train/test splitting, and evaluation |
| Matplotlib / Seaborn | Data visualization, if used to generate the project figures |
| Jupyter Notebook | Interactive analysis, if used in the project workflow |

---

## 🌷 Conclusion

This project demonstrates how supervised machine learning can be applied to an e-commerce business problem, from data preparation and model comparison to interpreting predictive patterns.

Among the three tested algorithms, **Random Forest achieved the strongest overall performance**, with 90.25% accuracy and an F1-score of 0.646.

The analysis also highlighted `PageValues` as the dominant predictive feature, followed by engagement and exit-related indicators. These findings illustrate how data analysis can help businesses better understand customer behavior and identify opportunities to improve conversion strategies.

The project also emphasizes the importance of looking beyond accuracy, particularly when the business outcome of interest represents a minority class.

---

<div align="center">

**Built with curiosity, data, and a little bit of pink. 💗**

*Academic Data Mining Project · Business Analytics*

</div>
