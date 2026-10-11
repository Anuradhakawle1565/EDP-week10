import os
import numpy as np
import pandas as pd

def load_close():
    path = "data/raw/AAPL_history.csv"
    if not os.path.exists(path):
        raise FileNotFoundError("Run download_data.py first.")
    df = pd.read_csv(path)
    df["Date"] = pd.to_datetime(df["Date"])
    df["Close"] = pd.to_numeric(df["Close"], errors="coerce")
    df = df.dropna(subset=["Date", "Close"]).sort_values("Date").reset_index(drop=True)
    return df[["Date", "Close"]]

def make_regression_data(df, lookback=5):
    rows = []
    closes = df["Close"].to_numpy()
    dates = df["Date"].to_numpy()
    for i in range(lookback, len(df)):
        row = {f"Close_t-{lookback-j}": closes[i-lookback+j] for j in range(lookback)}
        row["Target_Close"] = closes[i]
        row["Date"] = dates[i]
        rows.append(row)
    out = pd.DataFrame(rows)
    X = out[[c for c in out.columns if c.startswith("Close_t-")]]
    y = out["Target_Close"]
    return out, X, y

def make_lstm_data(df, lookback=60, train_ratio=0.8):
    values = df["Close"].to_numpy().reshape(-1, 1)
    split = int(len(values) * train_ratio)

    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler()
    scaler.fit(values[:split])

    scaled = scaler.transform(values)
    X, y, dates = [], [], []
    for i in range(lookback, len(scaled)):
        X.append(scaled[i-lookback:i, 0])
        y.append(scaled[i, 0])
        dates.append(df["Date"].iloc[i])

    X = np.array(X).reshape(-1, lookback, 1)
    y = np.array(y)
    dates = pd.to_datetime(dates)

    train_end = split - lookback
    return X[:train_end], y[:train_end], X[train_end:], y[train_end:], dates[train_end:], scaler

if __name__ == "__main__":
    df = load_close()
    os.makedirs("data/processed", exist_ok=True)
    reg, _, _ = make_regression_data(df)
    reg.to_csv("data/processed/regression_dataset.csv", index=False)
    print("Prepared regression dataset.")
