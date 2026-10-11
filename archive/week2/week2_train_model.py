# ==========================================
# STOCK PRICE TREND PREDICTOR
# WEEK 2 - LINEAR REGRESSION MODEL
# ==========================================

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
import joblib

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

# PROJECT SETTINGS
TICKER = "AAPL"
START_DATE = "2020-01-01"
END_DATE = "2026-01-01"

DATA_FOLDER = "data"
MODEL_FOLDER = "model"

os.makedirs(DATA_FOLDER, exist_ok=True)
os.makedirs(MODEL_FOLDER, exist_ok=True)

print("\n==========================================")
print(" STOCK PRICE TREND PREDICTOR")
print("==========================================")

# DOWNLOAD STOCK DATA
print("\nDownloading stock data...")

data = yf.download(
    TICKER,
    start=START_DATE,
    end=END_DATE,
    auto_adjust=False
)

if data.empty:
    print("ERROR: No stock data downloaded.")
    raise SystemExit

# CLEAN DATA
if isinstance(data.columns, pd.MultiIndex):
    data.columns = data.columns.get_level_values(0)

data = data.reset_index()
data = data[["Date", "Open", "High", "Low", "Close", "Volume"]]
data = data.dropna()

print("\nFirst 5 rows:")
print(data.head())

# SAVE RAW DATA
data.to_csv(
    os.path.join(DATA_FOLDER, f"{TICKER}_stock_data.csv"),
    index=False
)

# CREATE FEATURES
data["Previous_Close"] = data["Close"].shift(1)
data["Previous_Open"] = data["Open"].shift(1)
data["Previous_High"] = data["High"].shift(1)
data["Previous_Low"] = data["Low"].shift(1)
data["Previous_Volume"] = data["Volume"].shift(1)

data = data.dropna()

# INPUTS AND TARGET
X = data[
    [
        "Previous_Close",
        "Previous_Open",
        "Previous_High",
        "Previous_Low",
        "Previous_Volume"
    ]
]

y = data["Close"]

# TRAIN / TEST SPLIT
# shuffle=False is important for time-series data.
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    shuffle=False
)

print("\n==========================================")
print(" TRAIN / TEST SPLIT")
print("==========================================")
print("Total samples :", len(X))
print("Training data :", len(X_train))
print("Testing data  :", len(X_test))

# CREATE MODEL
model = LinearRegression()

# TRAIN MODEL
print("\nTraining Linear Regression model...")
model.fit(X_train, y_train)
print("Model training completed!")

# PREDICTIONS
y_pred = model.predict(X_test)

comparison = pd.DataFrame({
    "Actual Price": y_test.values,
    "Predicted Price": y_pred
})

print("\n==========================================")
print(" SAMPLE PREDICTIONS")
print("==========================================")
print(comparison.head(10))

comparison.to_csv(
    os.path.join(DATA_FOLDER, "actual_vs_predicted.csv"),
    index=False
)

# EVALUATION METRICS
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print("\n==========================================")
print(" MODEL EVALUATION")
print("==========================================")
print(f"Mean Absolute Error (MAE) : {mae:.4f}")
print(f"Mean Squared Error (MSE)  : {mse:.4f}")
print(f"Root Mean Squared Error   : {rmse:.4f}")
print(f"R2 Score                  : {r2:.4f}")

# COEFFICIENTS
print("\n==========================================")
print(" MODEL COEFFICIENTS")
print("==========================================")

coefficients = pd.DataFrame({
    "Feature": X.columns,
    "Coefficient": model.coef_
})

print(coefficients)
print("\nIntercept:", model.intercept_)

# SAVE MODEL
model_path = os.path.join(
    MODEL_FOLDER,
    "linear_regression_model.pkl"
)

joblib.dump(model, model_path)

print("\nModel saved successfully!")
print("Saved at:", model_path)

# ACTUAL VS PREDICTED GRAPH
plt.figure(figsize=(12, 6))

plt.plot(
    y_test.values,
    label="Actual Price"
)

plt.plot(
    y_pred,
    label="Predicted Price"
)

plt.title(f"{TICKER} Stock Price - Actual vs Predicted")
plt.xlabel("Test Data Points")
plt.ylabel("Closing Price")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("actual_vs_predicted.png", dpi=150)
plt.show()

# SCATTER PLOT
plt.figure(figsize=(8, 6))

plt.scatter(
    y_test,
    y_pred,
    alpha=0.6
)

minimum = min(y_test.min(), y_pred.min())
maximum = max(y_test.max(), y_pred.max())

plt.plot(
    [minimum, maximum],
    [minimum, maximum]
)

plt.title("Actual Price vs Predicted Price")
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.grid(True)
plt.tight_layout()
plt.savefig("actual_vs_predicted_scatter.png", dpi=150)
plt.show()

print("\n==========================================")
print(" WEEK 2 MODEL COMPLETED")
print("==========================================")
print("Stock             :", TICKER)
print("Algorithm         : Linear Regression")
print("Training samples  :", len(X_train))
print("Testing samples   :", len(X_test))
print(f"MAE               : {mae:.4f}")
print(f"MSE               : {mse:.4f}")
print(f"RMSE              : {rmse:.4f}")
print(f"R2 Score          : {r2:.4f}")
print("\nDone!")
