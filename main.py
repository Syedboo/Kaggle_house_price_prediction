train_df, test_df = load_data("train.csv", "test.csv")
y = train_df["SalePrice"]
X = preprocess_features(train_df.drop("SalePrice", axis=1))
numerical_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
categorical_cols = X.select_dtypes(include=["object"]).columns.tolist()
pipeline = build_pipeline(numerical_cols, categorical_cols)
model = train_model(pipeline, X, y)
predict_and_export(model, test_df)
