import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, FunctionTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error, r2_score
from xgboost import XGBRegressor


def load_data(train_path, test_path):
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    return train, test


def preprocess_features(df):
    df = df.drop(columns=["Id"], errors='ignore')
    return df


def build_pipeline(numerical_cols, categorical_cols):
    # Numeric pipeline
    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("log_transform", FunctionTransformer(np.log1p, validate=False)),
        ("scaler", StandardScaler())
    ])

    # Categorical pipeline
    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="constant", fill_value="Missing")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ])

    # Column transformer
    preprocessor = ColumnTransformer([
        ("num", numeric_pipeline, numerical_cols),
        ("cat", categorical_pipeline, categorical_cols)
    ])

    # Full pipeline
    model = Pipeline([
        ("preprocessor", preprocessor),
        ("regressor", XGBRegressor(n_estimators=500, learning_rate=0.05, max_depth=4, random_state=42))
    ])

    return model


def train_model(model, X, y):
    scores = cross_val_score(model, X, y, scoring='r2', cv=5)
    rmse_scores = cross_val_score(model, X, y, scoring='neg_root_mean_squared_error', cv=5)
    print("CV R^2 scores:", scores)
    print("Average R^2:", scores.mean())
    print("CV RMSE scores:", -rmse_scores)
    print("Average RMSE:", -rmse_scores.mean())
    model.fit(X, y)
    return model


def predict_and_export(model, test_df, output_path="submission.csv"):
    ids = test_df["Id"]
    test_df = preprocess_features(test_df)
    predictions = model.predict(test_df)
    output = pd.DataFrame({"Id": ids, "SalePrice": predictions})
    output.to_csv(output_path, index=False)
