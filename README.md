# Stock Price Trend Predictor — Week 10 Working Prototype

A working Streamlit prototype that brings together the core features of the Stock Price Trend Predictor: historical stock data, feature preparation, Linear Regression prediction, optional LSTM prediction, model evaluation, visualization, and moving-average backtesting.

> **Milestone:** Working prototype  
> **Week 10 focus:** Core capstone feature development  
> **Purpose:** Educational project, not financial advice.

## Features

- Download historical prices using Yahoo Finance (`yfinance`) or upload a CSV.
- Explore closing-price history.
- Engineer daily returns, moving averages, EMAs, volatility, lag features, and momentum.
- Train a Linear Regression model using chronological train/test splitting.
- Display MAE, RMSE and R².
- Show actual-vs-predicted prices and a next-step estimate.
- Optionally train an LSTM if TensorFlow is installed and supported.
- Backtest an MA20/MA50 moving-average strategy with transaction costs.
- Compare strategy equity with buy-and-hold.
- Download prediction and backtest results as CSV files.

## Folder structure

```text
Stock-Price-Trend-Predictor-Week10-Working-Prototype/
├── app.py
├── src/
│   └── prototype_core.py
├── archive/
│   ├── week2/
│   ├── week3/
│   ├── week4/
│   ├── week5/
│   └── week6-7/
├── data/
│   ├── raw/
│   └── processed/
├── models/
├── results/
├── docs/
├── requirements.txt
├── requirements-lstm.txt
├── .gitignore
└── README.md
```

`archive/` preserves the weekly source modules from the project package you provided. The integrated prototype uses `app.py` and `src/prototype_core.py`.

## Requirements

- Python supported by the installed versions of Streamlit, Pandas, scikit-learn and yfinance
- Internet connection for downloading data
- Or a CSV file with at least `Date` and `Close` columns

The LSTM option is optional because TensorFlow support varies by Python version and platform. On Python 3.14, TensorFlow may not be available for your environment; the core Linear Regression prototype does not require TensorFlow.

## Run in VS Code (Windows PowerShell)

1. Extract the ZIP.
2. Open the extracted folder in VS Code.
3. Open Terminal → New Terminal.
4. Confirm the terminal is at the project root (the folder containing `app.py` and `requirements.txt`).
5. Install packages:

```powershell
python -m pip install -r requirements.txt
```

6. Start the prototype:

```powershell
python -m streamlit run app.py
```

Streamlit will print a local URL (usually `http://localhost:8501`) in the terminal. Open that URL in your browser.

### If `streamlit` is not recognized

Use the command above with `python -m streamlit`. If Streamlit is not installed, rerun the requirements installation.

## Use the prototype

### Option A: Download stock data
- In the sidebar, choose **Download historical stock data**.
- Enter a ticker such as `AAPL` or `MSFT`.
- Select start and end dates.
- Click **Load stock data**.

### Option B: Upload your own CSV
The CSV must contain a `Date` column and a `Close` or `Adj Close` column. `Open`, `High`, `Low`, and `Volume` are optional. Use daily data and preferably at least 100 rows.

### Prediction
- Select Linear Regression.
- Review MAE, RMSE, R² and the next-step estimated close.
- Inspect the actual-vs-predicted chart.
- Download predictions as CSV.

### Backtesting
- Open the Backtesting tab.
- Choose transaction cost and initial capital.
- Compare the moving-average strategy against buy-and-hold.
- Download the backtest table.

## What the prototype saves

Generated artifacts are written locally:
- `data/processed/prototype_predictions.csv`
- `data/processed/prototype_lstm_predictions.csv` (if LSTM runs)
- `models/linear_regression_prototype.pkl`
- `models/prototype_lstm.keras` (if LSTM runs)
- `results/prototype_model_metrics.csv`
- `results/prototype_backtest_results.csv`
- `results/prototype_backtest_summary.csv`

Generated data, models and result files are ignored by Git to avoid uploading large or reproducible artifacts. If you need to share a small example output for a report, add a separate sanitized sample file intentionally.

## Week 10 milestone checklist

- [x] Integrate a user interface
- [x] Load stock data from online source or CSV
- [x] Prepare features and train a regression model
- [x] Display evaluation metrics and prediction graph
- [x] Add moving-average backtesting and comparison chart
- [x] Save downloadable CSV outputs
- [ ] Run end-to-end on your own computer and capture screenshots
- [ ] Confirm data-provider availability and verify outputs locally
- [ ] Optionally enable LSTM with a TensorFlow-compatible environment

## Week 10 skills learned

**Core feature integration, Streamlit application development, historical stock-data handling, feature engineering, model training, evaluation, data visualization, backtesting, debugging, and prototype testing.**

## Disclaimer

This is an educational prototype. Historical prices, model predictions and backtesting results do not guarantee future performance. Do not use this project alone to make investment decisions.
