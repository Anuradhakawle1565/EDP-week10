import os, time
import pandas as pd
import yfinance as yf

TICKERS=['AAPL','MSFT','GOOGL','AMZN','META','NVDA','TSLA','NFLX','JPM','V']
START_DATE='2015-01-01'; END_DATE='2026-08-18'
OUT='data/multi_stock_raw.csv'; os.makedirs('data',exist_ok=True)
all_data=[]
print('='*60); print('WEEK 4 - MULTI-STOCK FINANCIAL DATA'); print('='*60)
for ticker in TICKERS:
    print(f'Downloading {ticker}...')
    try:
        df=yf.download(ticker,start=START_DATE,end=END_DATE,auto_adjust=False,progress=False)
        if df.empty: print('  No data returned'); continue
        if isinstance(df.columns,pd.MultiIndex): df.columns=df.columns.get_level_values(0)
        df=df.reset_index(); needed=['Date','Open','High','Low','Close','Adj Close','Volume']
        df=df[[c for c in needed if c in df.columns]].copy(); df['Ticker']=ticker
        df=df.dropna(); all_data.append(df); print(f'  {len(df):,} rows')
    except Exception as e: print(f'  Error: {e}')
    time.sleep(.4)
if not all_data: raise RuntimeError('No data downloaded. Check internet connection/ticker symbols.')
data=pd.concat(all_data,ignore_index=True); data['Date']=pd.to_datetime(data['Date']); data=data.sort_values(['Ticker','Date']).reset_index(drop=True)
data.to_csv(OUT,index=False)
print(f'\nStocks: {data.Ticker.nunique()} | Rows: {len(data):,} | From: {data.Date.min().date()} | To: {data.Date.max().date()}'); print(f'Saved: {OUT}')
print(data.groupby('Ticker').size())
