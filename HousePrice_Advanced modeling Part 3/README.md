# 🏠 House Price - Advanced Modeling

## 📌 Project Overview

This project focuses on building and evaluating multiple supervised machine learning models to classify house prices using the Ames Housing dataset. The project covers data preprocessing, feature engineering, ensemble learning, hyperparameter tuning, cross-validation, feature importance analysis, learning curve evaluation, and model serialization. The objective is to identify the best-performing model for accurate and reliable house price classification.

---

# 📂 Project Structure

```
HousePrice_Advanced_Modeling_Part3/
│
├── README.md   ← Create here
├── best_model.pkl
│
├── data/
│   ├── train.csv
│   ├── test.csv
│   ├── learning_curve.csv
│   └── data_description.txt
│
├── images/
│   └── top5_feature_importance.png
│
└── notebooks/
    └── Part3_Advanced_Modeling.ipynb
```

 ## Folder Description

- *README.md* – Contains the project overview, implementation details, observations, and model comparison.
- *best_model.pkl* – Serialized best-performing machine learning pipeline saved using Joblib.
- *data/* – Contains the training dataset, testing dataset, learning curve results, and dataset description.
- *images/* – Contains the Top 5 Feature Importance plot generated from the Random Forest model.
- *notebooks/* – Contains the Jupyter Notebook with the complete implementation of the advanced modeling workflow.

# 🛠️ Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Scikit-Learn
- Joblib
- Jupyter Notebook

---

# ▶️ How to Run the Project

1. Clone this repository.
2. Install the required libraries.

```bash
pip install pandas numpy matplotlib scikit-learn joblib
```

3. Open Jupyter Notebook.

```bash
jupyter notebook
```

4. Open `Part3_Advanced_Modeling.ipynb`.

5. Run all cells from top to bottom.

---

# 🎯 Project Objective

The objective of this project is to compare multiple machine learning algorithms for binary house price classification and identify the best-performing model based on ROC-AUC score, cross-validation performance, and generalization ability. The project also demonstrates feature importance analysis, feature ablation, hyperparameter tuning, learning curve analysis, and model deployment using model serialization.

---

# 📊 Dataset

**Dataset:** Housing Dataset

**Target Variable:** `SalePrice_Category`

- **0** → Low Price
- **1** → High Price

# 🚀 How to Run the Project

## Prerequisites

- Python 3.x
- Jupyter Notebook
- Required libraries:
  - pandas
  - numpy
  - matplotlib
  - scikit-learn
  - joblib

## Installation

```bash
pip install pandas numpy matplotlib scikit-learn joblib
```

## Run the Project

1. Clone the repository.
2. Open the `notebooks/Part3_Advanced_Modeling.ipynb` notebook.
3. Run all cells from top to bottom.
4. The notebook will:
   - Train multiple machine learning models.
   - Compare model performance.
   - Perform feature importance analysis.
   - Execute feature selection.
   - Perform hyperparameter tuning using GridSearchCV.
   - Generate the learning curve.
   - Save the best model as `best_model.pkl`.

# Task 1: Decision Tree Constraints

## Objective
Compare an unconstrained Decision Tree with a constrained Decision Tree to understand the effect of limiting tree complexity on model performance.

## Method
- Trained an unconstrained Decision Tree classifier.
- Trained a constrained Decision Tree by limiting the maximum depth.
- Compared training accuracy, testing accuracy, and ROC-AUC.

## Observation
The unconstrained Decision Tree achieved very high training performance but showed a larger gap between training and testing results, indicating overfitting. The constrained Decision Tree reduced this gap and generalized better on unseen data. Limiting the tree depth helped improve the model's robustness by reducing variance.

# Task 2: Controlled Decision Tree

## Objective

Train a Decision Tree Classifier with controlled hyperparameters (`max_depth = 5` and `min_samples_split = 20`) to reduce overfitting and improve the model's ability to generalize to unseen data.

### Model Configuration

- **max_depth:** 5
- **min_samples_split:** 20
- **random_state:** 42

### Results

| Metric | Value |
|---------|------:|
| Training Accuracy | **93.84%** |
| Testing Accuracy | **90.41%** |

### Observation

The Controlled Decision Tree achieved a **Training Accuracy of 93.84%** and a **Testing Accuracy of 90.41%**. Compared to the baseline Decision Tree, the gap between the training and testing accuracy decreased, indicating reduced overfitting and improved generalization.

The **max_depth** parameter limits how deep the tree can grow, preventing the model from learning unnecessary details in the training data. The **min_samples_split** parameter ensures that a node is split only when it contains at least 20 samples, reducing unnecessary splits and improving model robustness.

## Task 3: Gini vs Entropy

### Objective
Compare Decision Tree models using the **Gini** and **Entropy** splitting criteria to understand their impact on model performance.

### Gini Impurity

The Gini Impurity measures the probability of incorrectly classifying a randomly selected sample if it is assigned a class label according to the class distribution in the node.

**Formula:**

$begin:math:display$
Gini \= 1 \- \\sum\_\{i\=1\}\^\{n\} p\_i\^2
$end:math:display$

where $begin:math:text$p\_i$end:math:text$ is the probability of class $begin:math:text$i$end:math:text$.

### Entropy

Entropy measures the amount of uncertainty or randomness in the data. The Decision Tree chooses splits that maximize information gain.

**Formula:**

$begin:math:display$
Entropy \= \-\\sum\_\{i\=1\}\^\{n\} p\_i \\log\_2\(p\_i\)
$end:math:display$

where $begin:math:text$p\_i$end:math:text$ is the probability of class $begin:math:text$i$end:math:text$.

### Observation

Both Gini and Entropy produced similar predictive performance on this dataset. Gini is computationally faster because it does not require logarithmic calculations, while Entropy is based on information theory. In this project, the performance difference between the two criteria was minimal.

## Task 4: Random Forest Classifier

### Objective

Train a Random Forest Classifier, evaluate its performance using Training Accuracy, Testing Accuracy, and ROC-AUC, and analyze the most important features influencing house price classification.

### Model Configuration

- **n_estimators:** 100
- **max_depth:** 10
- **random_state:** 42

### Results

| Metric | Value |
|---------|------:|
| Training Accuracy | **99.74%** |
| Testing Accuracy | **92.81%** |
| ROC-AUC | **0.9796** |

### Top 5 Important Features

| Feature | Importance |
|---------|-----------:|
| GrLivArea | 0.067951 |
| YearBuilt | 0.059878 |
| FullBath | 0.055129 |
| OverallQual | 0.054327 |
| GarageCars | 0.051691 |

### Feature Importance

Random Forest computes feature importance by measuring the **average reduction in Gini impurity** contributed by each feature across all decision trees in the forest. Features that consistently produce better splits receive higher importance scores.

Unlike linear regression coefficients, feature importance does **not** indicate the direction (positive or negative) of a feature's relationship with the target variable. Instead, it measures how useful a feature is in reducing impurity during tree construction.

### Bagging (Bootstrap Aggregating)

Random Forest is an ensemble learning algorithm that uses **bagging**. Each decision tree is trained on a **bootstrap sample**, which is created by randomly selecting observations from the training data **with replacement**. Additionally, at every split, only a random subset of approximately **√(number of features)** is considered.

This randomness creates diverse decision trees. The final prediction is obtained through majority voting, which reduces variance, minimizes overfitting, and improves generalization compared to a single Decision Tree.

### Observation

The Random Forest classifier achieved excellent predictive performance with a **Training Accuracy of 99.74%**, **Testing Accuracy of 92.81%**, and a **ROC-AUC score of 0.9796**. The model generalized well on unseen data and outperformed the Decision Tree models while reducing overfitting through ensemble learning.

## Task 4a: Gradient Boosting Classifier

### Objective

Train a Gradient Boosting Classifier and evaluate its performance using Training Accuracy, Testing Accuracy, and ROC-AUC score.

### Model Configuration

- **n_estimators:** 100
- **learning_rate:** 0.1
- **max_depth:** 3
- **random_state:** 42

### Results

| Metric | Value |
|---------|------:|
| Training Accuracy | **99.32%** |
| Testing Accuracy | **93.49%** |
| ROC-AUC | **0.9816** |

### Observation

The Gradient Boosting Classifier achieved excellent predictive performance with a **Training Accuracy of 99.32%**, **Testing Accuracy of 93.49%**, and a **ROC-AUC score of 0.9816**. Gradient Boosting builds decision trees sequentially, where each new tree focuses on correcting the errors made by previous trees. This sequential learning process improved predictive performance and produced slightly better test performance than the Random Forest model.

## Task 4b: Feature Ablation Study

### Objective

Evaluate the impact of removing the five least important features from the Random Forest model and compare its predictive performance with the original model.

### Five Least Important Features

- Condition2_PosN
- Neighborhood_BrDale
- Condition2_RRNn
- MiscFeature_TenC
- Exterior2nd_Other

### ROC-AUC Comparison

| Model | ROC-AUC |
|--------|---------:|
| Original Random Forest | **0.9796** |
| Reduced Random Forest | **0.9770** |

### Observation

The Random Forest model trained with all features achieved a ROC-AUC score of **0.9796**, while the model trained after removing the five least important features achieved a ROC-AUC score of **0.9770**.

The very small reduction in ROC-AUC indicates that these five features contributed little to the model's predictive performance. Removing uninformative features can simplify the model, reduce computational complexity, and improve maintainability while preserving almost the same predictive accuracy.

# Task 5: Cross-Validated Model Comparison

## Objective

Compare the performance of multiple machine learning models using 5-fold Stratified Cross-Validation with ROC-AUC as the evaluation metric.

### Cross-Validation Results

| Model | Mean ROC-AUC | Standard Deviation |
|--------|-------------:|-------------------:|
| Logistic Regression | **0.9514** | **0.0125** |
| Decision Tree | **0.9296** | **0.0107** |
| Random Forest | **0.9807** | **0.0056** |
| Gradient Boosting | **0.9787** | **0.0073** |

### Observation

Random Forest achieved the highest mean ROC-AUC (**0.9807**) while also having the lowest standard deviation (**0.0056**), indicating both excellent predictive performance and consistent generalization across the five folds. Gradient Boosting also performed exceptionally well but showed slightly higher variability. Logistic Regression provided a strong baseline, whereas the Decision Tree had the lowest overall performance.

---

# Task 6: Hyperparameter Tuning using GridSearchCV

## Objective

Optimize the Random Forest classifier by tuning its hyperparameters using GridSearchCV with 5-fold cross-validation.

### Parameter Grid

- n_estimators = [50, 100, 200]
- max_depth = [5, 10, None]
- min_samples_leaf = [1, 5]

### Best Parameters

- **n_estimators:** 200
- **max_depth:** 10
- **min_samples_leaf:** 1

### Best Cross-Validation ROC-AUC

**0.9812**

### Total Model Configurations

- Parameter combinations evaluated: **18**
- 5-fold cross-validation: **90 total model fits**

### GridSearchCV vs RandomizedSearchCV

GridSearchCV evaluates every possible hyperparameter combination within the specified search space, guaranteeing the best combination from the provided values. RandomizedSearchCV evaluates only a randomly selected subset of combinations, making it faster for large search spaces but without guaranteeing the optimal solution.

### Observation

The optimized Random Forest model achieved a cross-validated ROC-AUC of **0.9812**, demonstrating a slight improvement over the default configuration.

---

# Task 7: Manual Learning Curve

## Objective

Evaluate model performance by training the optimized model on increasing fractions of the training dataset.

### Learning Curve Results

| Training Fraction | Training ROC-AUC | Test ROC-AUC |
|------------------:|-----------------:|-------------:|
| 20% | 1.0000 | 0.9796 |
| 40% | 1.0000 | 0.9795 |
| 60% | 1.0000 | 0.9779 |
| 80% | 0.9999 | 0.9813 |
| 100% | 0.9999 | 0.9806 |

### Observation

Training ROC-AUC remained close to **1.0** across all training fractions, while the Test ROC-AUC remained consistently high. The performance stabilized between **80% and 100%** of the training data, indicating that the model has largely reached its learning capacity. The model appears to be **capacity-limited rather than data-limited**, suggesting that collecting significantly more data is unlikely to provide substantial performance improvements.

---

# Task 8: Model Serialization

## Objective

Save the optimized model using Joblib, reload it, and verify that it can make predictions successfully.

### Observation

The optimized Random Forest pipeline was successfully saved as **best_model.pkl** using Joblib. The serialized model was reloaded without errors and produced correct predictions on unseen test samples, confirming that the model can be reused for future inference without retraining.

---

# Task 9: Final Model Comparison

| Model | CV Mean ROC-AUC | CV Std ROC-AUC |
|--------|----------------:|---------------:|
| Logistic Regression | 0.9514 | 0.0125 |
| Decision Tree | 0.9296 | 0.0107 |
| Random Forest | **0.9807** | **0.0056** |
| Gradient Boosting | 0.9787 | 0.0073 |

## Final Recommendation

Among all the evaluated models, the **Random Forest Classifier** is recommended as the final model. It achieved the highest average cross-validation ROC-AUC while maintaining the lowest variability across folds, demonstrating excellent predictive performance and strong generalization. Although Gradient Boosting achieved a slightly higher test ROC-AUC, Random Forest provided the best balance between accuracy, robustness, and stability.

---

# Conclusion

This project successfully demonstrated an end-to-end machine learning workflow for house price classification. Multiple supervised learning models were trained, evaluated, and compared using appropriate performance metrics. Ensemble methods significantly outperformed individual Decision Tree models, with Random Forest emerging as the most reliable model after hyperparameter tuning. The project also covered feature importance analysis, feature ablation, cross-validation, learning curve evaluation, and model serialization, providing a comprehensive understanding of advanced machine learning techniques.