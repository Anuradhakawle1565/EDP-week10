import os
import yfinance as yf

TICKER = "AAPL"
START = "2015-01-01"
END = "2026-08-18"

def download():
    os.makedirs("data/raw", exist_ok=True)
    df = yf.download(TICKER, start=START, end=END, auto_adjust=False, progress=False)
    if df.empty:
        raise RuntimeError("No data downloaded. Check your internet connection or ticker.")
    if hasattr(df.columns, "levels"):
        df.columns = df.columns.get_level_values(0)
    df = df.reset_index()
    df.to_csv("data/raw/AAPL_history.csv", index=False)
    print(f"Downloaded {len(df)} rows for {TICKER}.")
    return df

if __name__ == "__main__":
    download()
