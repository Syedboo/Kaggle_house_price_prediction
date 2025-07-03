# Kaggle_house_price_prediction

# 🏠 House Price Prediction using XGBoost and ML Pipeline

This project implements a robust machine learning pipeline to predict house prices using the [Kaggle House Prices - Advanced Regression Techniques](https://www.kaggle.com/c/house-prices-advanced-regression-techniques) dataset. The solution is built using a modular `scikit-learn` pipeline integrated with `XGBoost`, allowing clean handling of preprocessing, transformations, and model training.

---

## 🚀 Result Summary

- ✅ **Leaderboard Score:** `0.12688` (Log RMSE)
- 🏅 **Public Ranking:** Top ~**20%** (Rank 900 out of 5400+ submissions)
- 🧪 First attempt submission

---

## 📁 Project Structure
├── pipeline.py # Main ML pipeline code
├── main.py # Script to train and generate submission
├── requirements.txt # Python dependencies
├── submission.csv # Sample output
├── train.csv # Training data
├── test.csv # Testing data
├── data_description # Crucial for EDA and Preprocessing
└── README.md # This file


---

## 🛠️ Machine Learning Workflow

1. 📦 Data Loading

```python
train_df, test_df = load_data("train.csv", "test.csv")


**### 2. 🧼 Data Preprocessing**
Dropped unnecessary features (Id)

Filled missing numeric values with median

Categorical features filled with "Missing" and One-Hot Encoded

Handled skewed numeric features using log1p transformation

**### 3. 🔧 Pipeline and Model**
Combined numeric and categorical pipelines using ColumnTransformer

Used XGBRegressor with:

  XGBRegressor(n_estimators=500, learning_rate=0.05, max_depth=4)


**### 4. 📊 Model Evaluation**
5-fold cross-validation results:

Average R²: ~0.89

Average RMSE: ~25,900

**### 5. 📤 Submission File**
Generates a valid Kaggle submission:
Id,SalePrice
1461,208500.0
1462,181500.0
...

