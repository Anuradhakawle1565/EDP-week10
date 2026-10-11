import os
import joblib
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def train():
    df = pd.read_csv("data/processed/regression_dataset.csv")
    feature_cols = [c for c in df.columns if c.startswith("Close_t-")]
    X, y = df[feature_cols], df["Target_Close"]

    split = int(len(df) * 0.8)
    X_train, X_test = X.iloc[:split], X.iloc[split:]
    y_train, y_test = y.iloc[:split], y.iloc[split:]
    dates = pd.to_datetime(df["Date"].iloc[split:])

    model = LinearRegression()
    model.fit(X_train, y_train)
    pred = model.predict(X_test)

    mae = mean_absolute_error(y_test, pred)
    rmse = mean_squared_error(y_test, pred) ** 0.5
    r2 = r2_score(y_test, pred)

    os.makedirs("models", exist_ok=True)
    joblib.dump(model, "models/linear_regression.pkl")

    out = pd.DataFrame({"Date": dates, "Actual": y_test.values, "Predicted": pred})
    os.makedirs("data/processed", exist_ok=True)
    out.to_csv("data/processed/regression_predictions.csv", index=False)

    print(f"Linear Regression -> MAE: {mae:.4f}, RMSE: {rmse:.4f}, R2: {r2:.4f}")
    return {"Model": "Linear Regression", "MAE": mae, "RMSE": rmse, "R2": r2}, out

if __name__ == "__main__":
    train()
