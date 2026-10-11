from pathlib import Path
import os
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import MinMaxScaler

ROOT = Path(__file__).resolve().parents[1]
DATA_RAW = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"
MODELS = ROOT / "models"
RESULTS = ROOT / "results"

for folder in [DATA_RAW, DATA_PROCESSED, MODELS, RESULTS]:
    folder.mkdir(parents=True, exist_ok=True)


def normalize_uploaded_data(df):
    if df is None or df.empty:
        raise ValueError("The uploaded CSV is empty.")
    # Normalize common column labels.
    lookup = {str(c).strip().lower().replace("_", " "): c for c in df.columns}
    date_col = lookup.get("date") or lookup.get("datetime")
    close_col = lookup.get("close") or lookup.get("adj close") or lookup.get("adjusted close")
    if date_col is None or close_col is None:
        raise ValueError("CSV must contain a Date column and a Close (or Adj Close) column.")
    out = df.copy()
    out = out.rename(columns={date_col: "Date", close_col: "Close"})
    out["Date"] = pd.to_datetime(out["Date"], errors="coerce")
    out["Close"] = pd.to_numeric(out["Close"], errors="coerce")
    for col in ["Open", "High", "Low", "Volume"]:
        if col not in out.columns:
            out[col] = np.nan
        else:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    out = out.dropna(subset=["Date", "Close"]).sort_values("Date").drop_duplicates("Date")
    out = out[["Date", "Open", "High", "Low", "Close", "Volume"]].reset_index(drop=True)
    if out.empty:
        raise ValueError("No valid Date/Close rows were found.")
    return out


def fetch_stock_data(ticker, start, end):
    try:
        import yfinance as yf
    except ImportError as exc:
        raise RuntimeError("yfinance is missing. Install it with: python -m pip install yfinance") from exc
    df = yf.download(ticker, start=start, end=end, auto_adjust=False, progress=False)
    if df is None or df.empty:
        raise ValueError("No data returned. Check ticker, date range, or internet connection.")
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df.reset_index()
    # yfinance may return Adj Close or not; Close is required.
    return normalize_uploaded_data(df)


def add_features(df):
    out = normalize_uploaded_data(df)
    out["Daily_Return_%"] = out["Close"].pct_change() * 100
    for window in [5, 10, 20, 50]:
        out[f"MA_{window}"] = out["Close"].rolling(window).mean()
    out["EMA_12"] = out["Close"].ewm(span=12, adjust=False).mean()
    out["EMA_26"] = out["Close"].ewm(span=26, adjust=False).mean()
    out["Volatility_20"] = out["Daily_Return_%"].rolling(20).std()
    for lag in range(1, 6):
        out[f"Close_Lag_{lag}"] = out["Close"].shift(lag)
    out["Volume_Change_%"] = out["Volume"].pct_change() * 100
    out["Momentum_10"] = out["Close"] - out["Close"].shift(10)
    return out


def _model_frame(df):
    frame = add_features(df)
    lag_cols = [f"Close_Lag_{lag}" for lag in range(1, 6)]
    frame["Target"] = frame["Close"]
    frame = frame.dropna(subset=lag_cols + ["Target"]).reset_index(drop=True)
    return frame, lag_cols


def train_regression(df, test_percent=20):
    frame, feature_cols = _model_frame(df)
    if len(frame) < 30:
        raise ValueError("Not enough valid rows after feature preparation. Use a longer date range.")
    split = int(len(frame) * (1 - test_percent / 100))
    split = max(10, min(split, len(frame) - 1))
    X_train, X_test = frame[feature_cols].iloc[:split], frame[feature_cols].iloc[split:]
    y_train, y_test = frame["Target"].iloc[:split], frame["Target"].iloc[split:]
    model = LinearRegression()
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    metrics = {
        "MAE": float(mean_absolute_error(y_test, pred)),
        "RMSE": float(mean_squared_error(y_test, pred) ** 0.5),
        "R2": float(r2_score(y_test, pred)) if len(y_test) > 1 else float("nan"),
    }
    pred_df = pd.DataFrame({
        "Date": frame["Date"].iloc[split:].values,
        "Actual": y_test.values,
        "Predicted": pred,
    })
    try:
        import joblib
        joblib.dump(model, MODELS / "linear_regression_prototype.pkl")
    except Exception:
        pass
    pred_df.to_csv(DATA_PROCESSED / "prototype_predictions.csv", index=False)
    pd.DataFrame([metrics]).to_csv(RESULTS / "prototype_model_metrics.csv", index=False)
    return metrics, pred_df, model, feature_cols


def predict_next_close(model, df, feature_cols):
    frame, _ = _model_frame(df)
    last = frame[feature_cols].iloc[[-1]]
    return float(model.predict(last)[0])


def run_backtest(df, transaction_cost_bps=10, initial_capital=10000.0):
    out = normalize_uploaded_data(df)
    out["MA20"] = out["Close"].rolling(20).mean()
    out["MA50"] = out["Close"].rolling(50).mean()
    out["Signal"] = (out["MA20"] > out["MA50"]).astype(float)
    out["Position"] = out["Signal"].shift(1).fillna(0.0)
    out["Market_Return"] = out["Close"].pct_change().fillna(0.0)
    out["Turnover"] = out["Position"].diff().abs().fillna(out["Position"].abs())
    cost = transaction_cost_bps / 10000.0
    out["Strategy_Return"] = out["Position"] * out["Market_Return"] - out["Turnover"] * cost
    out["BuyHold_Return"] = out["Market_Return"]
    out["Strategy Equity"] = initial_capital * (1 + out["Strategy_Return"]).cumprod()
    out["BuyHold Equity"] = initial_capital * (1 + out["BuyHold_Return"]).cumprod()
    out["Strategy Drawdown"] = out["Strategy Equity"] / out["Strategy Equity"].cummax() - 1
    out["BuyHold Drawdown"] = out["BuyHold Equity"] / out["BuyHold Equity"].cummax() - 1
    out.to_csv(RESULTS / "prototype_backtest_results.csv", index=False)
    return out


def calculate_metrics(bt, initial_capital=10000.0):
    strategy_final = float(bt["Strategy Equity"].iloc[-1])
    buyhold_final = float(bt["BuyHold Equity"].iloc[-1])
    n = max(len(bt), 1)
    years = max(n / 252.0, 1 / 252.0)
    strategy_returns = bt["Strategy_Return"].dropna()
    volatility = float(strategy_returns.std(ddof=1) * np.sqrt(252)) if len(strategy_returns) > 1 else 0.0
    sharpe = float((strategy_returns.mean() / strategy_returns.std(ddof=1)) * np.sqrt(252)) if len(strategy_returns) > 1 and strategy_returns.std(ddof=1) > 0 else 0.0
    metrics = {
        "Initial Capital": round(float(initial_capital), 2),
        "Strategy Final Value": round(strategy_final, 2),
        "Strategy Total Return (%)": round((strategy_final / initial_capital - 1) * 100, 2),
        "Buy & Hold Final Value": round(buyhold_final, 2),
        "Buy & Hold Total Return (%)": round((buyhold_final / initial_capital - 1) * 100, 2),
        "Strategy Annualized Volatility (%)": round(volatility * 100, 2),
        "Strategy Sharpe Ratio": round(sharpe, 3),
        "Strategy Max Drawdown (%)": round(float(bt["Strategy Drawdown"].min()) * 100, 2),
        "Buy & Hold Max Drawdown (%)": round(float(bt["BuyHold Drawdown"].min()) * 100, 2),
        "Position Changes": int(bt["Turnover"].sum()),
    }
    pd.DataFrame([metrics]).to_csv(RESULTS / "prototype_backtest_summary.csv", index=False)
    return metrics


def train_lstm(df, test_percent=20):
    try:
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import LSTM, Dense, Dropout
        from tensorflow.keras.callbacks import EarlyStopping
    except Exception as exc:
        raise ImportError("TensorFlow is unavailable or incompatible with this Python environment.") from exc

    raw = normalize_uploaded_data(df)
    values = raw["Close"].values.reshape(-1, 1)
    split_index = int(len(values) * (1 - test_percent / 100))
    if split_index <= 65 or len(values) - split_index < 5:
        raise ValueError("Need more data for LSTM. Use at least 100 daily observations.")
    scaler = MinMaxScaler()
    scaler.fit(values[:split_index])
    scaled = scaler.transform(values)
    lookback = 60
    X, y, dates = [], [], []
    for i in range(lookback, len(scaled)):
        X.append(scaled[i-lookback:i, 0])
        y.append(scaled[i, 0])
        dates.append(raw["Date"].iloc[i])
    X = np.array(X).reshape(-1, lookback, 1)
    y = np.array(y)
    dates = pd.to_datetime(dates)
    train_end = split_index - lookback
    X_train, y_train = X[:train_end], y[:train_end]
    X_test, y_test = X[train_end:], y[train_end:]
    test_dates = dates[train_end:]
    if len(X_train) < 10 or len(X_test) < 1:
        raise ValueError("Not enough rows for LSTM train/test split.")

    model = Sequential([
        LSTM(32, input_shape=(lookback, 1)),
        Dropout(0.1),
        Dense(1)
    ])
    model.compile(optimizer="adam", loss="mse")
    model.fit(
        X_train, y_train, epochs=12, batch_size=32,
        validation_split=0.1, callbacks=[EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True)],
        verbose=0
    )
    pred_scaled = model.predict(X_test, verbose=0)
    pred = scaler.inverse_transform(pred_scaled).ravel()
    actual = scaler.inverse_transform(y_test.reshape(-1, 1)).ravel()
    metrics = {
        "MAE": float(mean_absolute_error(actual, pred)),
        "RMSE": float(mean_squared_error(actual, pred) ** 0.5),
        "R2": float(r2_score(actual, pred)) if len(actual) > 1 else float("nan"),
    }
    pred_df = pd.DataFrame({"Date": test_dates, "Actual": actual, "Predicted": pred})
    model.save(MODELS / "prototype_lstm.keras")
    pred_df.to_csv(DATA_PROCESSED / "prototype_lstm_predictions.csv", index=False)
    pd.DataFrame([metrics]).to_csv(RESULTS / "prototype_lstm_metrics.csv", index=False)
    next_scaled = model.predict(scaled[-lookback:].reshape(1, lookback, 1), verbose=0)
    next_pred = float(scaler.inverse_transform(next_scaled)[0, 0])
    return metrics, pred_df, next_pred
