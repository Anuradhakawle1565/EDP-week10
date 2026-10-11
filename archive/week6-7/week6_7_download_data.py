import os, pandas as pd, yfinance as yf
TICKER='AAPL'; START_DATE='2018-01-01'; END_DATE='2026-08-18'
os.makedirs('data',exist_ok=True)
df=yf.download(TICKER,start=START_DATE,end=END_DATE,auto_adjust=False,progress=False)
if df.empty: raise RuntimeError('No data downloaded. Check internet connection or ticker.')
if isinstance(df.columns,pd.MultiIndex): df.columns=df.columns.get_level_values(0)
df=df.reset_index(); cols=['Date','Open','High','Low','Close','Adj Close','Volume']; df=df[[c for c in cols if c in df.columns]].dropna(); df['Date']=pd.to_datetime(df['Date'])
df.to_csv('data/backtest_prices.csv',index=False)
print(f'Downloaded {len(df):,} rows. Saved data/backtest_prices.csv')
