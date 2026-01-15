# Logistic Regression — Bank Churn Prediction

This project implements a complete **Logistic Regression model** on the Bank Churn dataset with preprocessing, EDA, model training, evaluation metrics, ROC-AUC analysis, cross-validation, and a bonus deep-learning version using Keras.

---

## Project Overview

This notebook includes:

- Dataset loading  
- Data cleaning  
- Missing value check  
- Feature Scaling  
- Train/Test split  
- Logistic Regression training  
- Confusion Matrix  
- ROC Curve + AUC Score  
- Classification Report  
- Cross Validation  
- Feature Importance  
- Bonus ANN using Keras  

---

## Repository Structure

```
Logistic-Regression-Bank-Churn-Prediction/
│
├── logistic_regression_bank_churn.ipynb
├── bank_churn.csv
└── README.md
```

---


## Dataset

Dataset used: **bank_churn.csv**

Main columns:

- ID  
- active_member  
- age  
- credit_score  
- estimated_salary  
- products_number  
- tenure  
- churn (target)

Dataset contains **no missing values**.

---

## Logistic Regression Model

1. Training code:

  ```python
  log_reg = LogisticRegression(max_iter=3000, solver='lbfgs')
  log_reg.fit(X_train_scaled, y_train)
  ```

2. Cross-validation:

  ```python
  cv_scores = cross_val_score(log_reg, X, y, cv=10, scoring='accuracy')
  ```

---


## ANN Model (Keras)

1. Model definition:

  ```python
  model = Sequential()
  model.add(Dense(1, input_dim=X_train.shape[1], activation='sigmoid'))
  ```

2. Training:

  ```python
  model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
  model.fit(X_train, y_train, validation_split=0.2, epochs=50, batch_size=32)
  ```

---

## Installation & Setup

1. Clone the repository:

  ```bash
  git clone <repo-link>
  cd logistic-regression-bank-churn-prediction
  ```

2. Upload notebook to Google Colab:

  ```bash
  logistic_regression_bank_churn.ipynb
  ```

3. Install required libraries:

  ```bash
  pip install numpy pandas scikit-learn matplotlib seaborn tensorflow
  ```

---

## How to Run

1. Open the notebook in Google Colab  
2. Upload `bank_churn.csv`  
3. Run all cells  
4. View evaluation metrics & plots  

---

## Evaluation Metrics

The notebook generates:

- Accuracy score  
- Confusion matrix  
- ROC curve  
- AUC value  
- Classification Report  
- Feature Importance  
- Cross-validation accuracy  

---



## Conclusion

This project demonstrates a complete ML workflow including preprocessing, Logistic Regression modeling, evaluation, cross-validation, and a deep-learning variant. It is suitable for academic submissions and practical ML demonstrations.

---

## Author

**Author:** Jairaj R  
GitHub: https://github.com/jairajrenjith
