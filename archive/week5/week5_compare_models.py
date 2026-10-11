import os
import pandas as pd
import matplotlib.pyplot as plt

def compare(reg_metrics, lstm_metrics, reg_df, lstm_df):
    metrics = pd.DataFrame([reg_metrics, lstm_metrics])
    os.makedirs("results", exist_ok=True)
    metrics.to_csv("results/model_metrics.csv", index=False)

    plt.figure(figsize=(12, 6))
    plt.plot(reg_df["Date"], reg_df["Actual"], label="Actual Price")
    plt.plot(reg_df["Date"], reg_df["Predicted"], label="Linear Regression")
    plt.plot(lstm_df["Date"], lstm_df["Predicted"], label="LSTM")
    plt.title("Stock Price Prediction: Regression vs LSTM")
    plt.xlabel("Date")
    plt.ylabel("Close Price")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("results/regression_vs_lstm.png", dpi=150)
    plt.close()

    print("\nModel Comparison:")
    print(metrics.to_string(index=False))

if __name__ == "__main__":
    print("Run main.py to train both models.")
