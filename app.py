import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

from prototype_core import (
    fetch_stock_data, normalize_uploaded_data, add_features,
    train_regression, predict_next_close, run_backtest, calculate_metrics
)

st.set_page_config(page_title="Stock Price Trend Predictor", page_icon="📈", layout="wide")

st.title("📈 Stock Price Trend Predictor")
st.caption("Week 10 Capstone Prototype · Historical data · Machine learning · Backtesting")
st.info("Educational prototype only. Predictions and historical backtests do not guarantee future market performance.")

with st.sidebar:
    st.header("Data Settings")
    data_mode = st.radio("Choose data source", ["Download historical stock data", "Upload CSV"])
    if data_mode == "Download historical stock data":
        ticker = st.text_input("Stock ticker", value="AAPL").strip().upper()
        start_date = st.date_input("Start date", value=pd.Timestamp("2018-01-01").date())
        end_date = st.date_input("End date", value=pd.Timestamp.today().date())
        load_clicked = st.button("Load stock data", type="primary", use_container_width=True)
        uploaded_file = None
    else:
        uploaded_file = st.file_uploader("Upload stock CSV", type=["csv"])
        load_clicked = st.button("Use uploaded data", type="primary", use_container_width=True)
        ticker = "Uploaded stock"
        start_date = None
        end_date = None

    st.divider()
    st.header("Model Settings")
    model_choice = st.selectbox("Prediction model", ["Linear Regression", "LSTM (optional TensorFlow)"])
    test_percent = st.slider("Test data (%)", min_value=10, max_value=40, value=20, step=5)
    st.caption("The data is split chronologically to respect time order.")

if "stock_data" not in st.session_state:
    st.session_state.stock_data = None
    st.session_state.ticker = None

if load_clicked:
    try:
        if data_mode == "Download historical stock data":
            if not ticker:
                st.error("Enter a stock ticker, for example AAPL or MSFT.")
            elif start_date >= end_date:
                st.error("The start date must be earlier than the end date.")
            else:
                with st.spinner(f"Downloading {ticker} historical data..."):
                    df = fetch_stock_data(ticker, str(start_date), str(end_date))
                st.session_state.stock_data = df
                st.session_state.ticker = ticker
                st.success(f"Loaded {len(df):,} rows for {ticker}.")
        else:
            if uploaded_file is None:
                st.warning("Choose a CSV file first.")
            else:
                df = normalize_uploaded_data(pd.read_csv(uploaded_file))
                st.session_state.stock_data = df
                st.session_state.ticker = ticker
                st.success(f"Loaded {len(df):,} rows from CSV.")
    except Exception as exc:
        st.error(f"Could not load data: {exc}")
        st.caption("Check your internet connection/ticker, or upload a CSV with Date and Close columns.")

df = st.session_state.stock_data

if df is None:
    st.markdown("""
    ### Welcome
    This prototype connects the main project features in one interface:
    1. Load historical prices from Yahoo Finance or upload your own CSV.
    2. Train a chronological Linear Regression model.
    3. View evaluation metrics and predicted-vs-actual prices.
    4. Generate a next-step closing-price estimate.
    5. Compare a moving-average strategy with buy-and-hold.

    **To begin:** choose a data source in the sidebar and click **Load stock data** or **Use uploaded data**.

    **CSV format:** include a `Date` column and a `Close` column. `Open`, `High`, `Low`, and `Volume` are optional.
    """)
    st.stop()

if len(df) < 70:
    st.error("Please use at least 70 daily observations so the model and moving-average strategy have enough data.")
    st.stop()

df = add_features(df)
ticker_label = st.session_state.ticker or "Stock"
latest_close = float(df["Close"].iloc[-1])
previous_close = float(df["Close"].iloc[-2])
change = latest_close - previous_close
change_pct = (change / previous_close * 100) if previous_close else 0

m1, m2, m3, m4 = st.columns(4)
m1.metric("Latest close", f"{latest_close:,.2f}", f"{change:+,.2f} ({change_pct:+.2f}%)")
m2.metric("Rows loaded", f"{len(df):,}")
m3.metric("First date", df["Date"].min().strftime("%Y-%m-%d"))
m4.metric("Last date", df["Date"].max().strftime("%Y-%m-%d"))

st.subheader(f"{ticker_label} · Historical closing price")
fig, ax = plt.subplots(figsize=(12, 4))
ax.plot(df["Date"], df["Close"], label="Close", linewidth=1.5)
ax.set_xlabel("Date")
ax.set_ylabel("Price")
ax.grid(alpha=0.25)
ax.legend()
fig.tight_layout()
st.pyplot(fig)
plt.close(fig)

tab_model, tab_backtest, tab_data = st.tabs(["🤖 Prediction Model", "🔁 Backtesting", "📄 Data Preview"])

with tab_model:
    st.subheader(f"{model_choice} results")
    if model_choice == "Linear Regression":
        try:
            metrics, pred_df, model, feature_cols = train_regression(df, test_percent=test_percent)
            a, b, c = st.columns(3)
            a.metric("MAE", f"{metrics['MAE']:.4f}")
            b.metric("RMSE", f"{metrics['RMSE']:.4f}")
            c.metric("R²", f"{metrics['R2']:.4f}")
            next_pred = predict_next_close(model, df, feature_cols)
            st.metric("Next-step estimated close", f"{next_pred:,.2f}")
            st.caption("This is a model estimate based on the latest available lag features, not a guaranteed future price.")
            fig, ax = plt.subplots(figsize=(12, 4))
            ax.plot(pred_df["Date"], pred_df["Actual"], label="Actual", linewidth=1.6)
            ax.plot(pred_df["Date"], pred_df["Predicted"], label="Predicted", linewidth=1.3)
            ax.set_title("Actual vs Predicted Close")
            ax.set_xlabel("Date")
            ax.set_ylabel("Price")
            ax.grid(alpha=0.25)
            ax.legend()
            fig.tight_layout()
            st.pyplot(fig)
            plt.close(fig)
            st.download_button(
                "Download prediction results CSV",
                pred_df.to_csv(index=False).encode("utf-8"),
                file_name="prediction_results.csv",
                mime="text/csv"
            )
        except Exception as exc:
            st.error(f"Could not train Linear Regression: {exc}")
    else:
        st.write("LSTM is an optional extension because TensorFlow support depends on your installed Python version.")
        try:
            from prototype_core import train_lstm
            with st.spinner("Training LSTM. This may take a few minutes..."):
                metrics, pred_df, next_pred = train_lstm(df, test_percent=test_percent)
            a, b, c = st.columns(3)
            a.metric("MAE", f"{metrics['MAE']:.4f}")
            b.metric("RMSE", f"{metrics['RMSE']:.4f}")
            c.metric("R²", f"{metrics['R2']:.4f}")
            st.metric("Next-step estimated close", f"{next_pred:,.2f}")
            fig, ax = plt.subplots(figsize=(12, 4))
            ax.plot(pred_df["Date"], pred_df["Actual"], label="Actual")
            ax.plot(pred_df["Date"], pred_df["Predicted"], label="LSTM predicted")
            ax.set_title("LSTM: Actual vs Predicted Close")
            ax.set_xlabel("Date")
            ax.set_ylabel("Price")
            ax.grid(alpha=0.25)
            ax.legend()
            fig.tight_layout()
            st.pyplot(fig)
            plt.close(fig)
            st.download_button(
                "Download LSTM results CSV",
                pred_df.to_csv(index=False).encode("utf-8"),
                file_name="lstm_prediction_results.csv",
                mime="text/csv"
            )
        except ImportError:
            st.warning("TensorFlow is not installed or is incompatible with this Python version. Use Linear Regression now, or install a Python version supported by TensorFlow and then install requirements-lstm.txt.")
        except Exception as exc:
            st.error(f"LSTM could not run: {exc}")

with tab_backtest:
    st.subheader("20-day / 50-day moving-average strategy")
    st.write("The signal is shifted by one trading day before returns are applied to reduce look-ahead bias. A transaction cost is applied when the position changes.")
    cost_bps = st.slider("Transaction cost per position change (basis points)", 0, 50, 10, 5)
    initial_capital = st.number_input("Initial capital", min_value=100.0, value=10000.0, step=1000.0)
    bt = run_backtest(df, transaction_cost_bps=cost_bps, initial_capital=initial_capital)
    metrics = calculate_metrics(bt, initial_capital)
    a, b, c = st.columns(3)
    a.metric("Strategy return", f"{metrics['Strategy Total Return (%)']:.2f}%")
    b.metric("Buy & hold return", f"{metrics['Buy & Hold Total Return (%)']:.2f}%")
    c.metric("Strategy max drawdown", f"{metrics['Strategy Max Drawdown (%)']:.2f}%")
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(bt["Date"], bt["Strategy Equity"], label="Moving-average strategy")
    ax.plot(bt["Date"], bt["BuyHold Equity"], label="Buy & hold")
    ax.set_title("Equity Curve")
    ax.set_xlabel("Date")
    ax.set_ylabel("Portfolio value")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)
    st.dataframe(pd.DataFrame([metrics]), use_container_width=True)
    st.download_button(
        "Download backtest results CSV",
        bt.to_csv(index=False).encode("utf-8"),
        file_name="backtest_results.csv",
        mime="text/csv"
    )

with tab_data:
    st.subheader("Prepared data")
    st.dataframe(df.tail(30), use_container_width=True)
    st.download_button(
        "Download loaded stock data CSV",
        df.to_csv(index=False).encode("utf-8"),
        file_name=f"{str(ticker_label).replace(' ', '_')}_stock_data.csv",
        mime="text/csv"
    )

st.divider()
st.caption("Week 10 milestone: working prototype integrating historical data, prediction, evaluation, visualization, and backtesting.")
