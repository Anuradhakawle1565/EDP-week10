# Week 10 — Core Capstone Feature & Working Prototype

## Goal
Integrate the central Stock Price Trend Predictor functions into a prototype that can be demonstrated end-to-end.

## Included
1. Historical stock-data loading by ticker or CSV upload.
2. Data preparation and feature engineering.
3. Linear Regression training and evaluation.
4. Optional LSTM training when TensorFlow is supported.
5. Actual-versus-predicted chart.
6. Moving-average backtesting with transaction-cost input.
7. Strategy-vs-buy-and-hold comparison.
8. Downloadable prediction/backtest CSVs.

## Demo acceptance checks
- Load a valid ticker and date range, or upload a CSV with Date and Close.
- Confirm the historical chart appears.
- Train Linear Regression and confirm MAE, RMSE, R², and prediction graph appear.
- Open Backtesting and confirm equity curve and metrics appear.
- Download the result CSVs.
- Capture screenshots for the weekly review.

## Known limitation
The LSTM option requires TensorFlow. TensorFlow compatibility depends on Python version and operating system. The main prototype works without TensorFlow.
