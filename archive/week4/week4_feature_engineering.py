import numpy as np, pandas as pd
INPUT='data/multi_stock_raw.csv'; OUTPUT='data/multi_stock_features.csv'
df=pd.read_csv(INPUT); df['Date']=pd.to_datetime(df['Date']); df=df.sort_values(['Ticker','Date']).reset_index(drop=True); g=df.groupby('Ticker',group_keys=False)
df['Price_Change']=df['Close']-df['Open']; df['Price_Change_%']=(df['Close']-df['Open'])/df['Open']*100
df['High_Low_Range']=df['High']-df['Low']; df['High_Low_Range_%']=(df['High']-df['Low'])/df['Close']*100
df['Open_Close_Range_%']=(df['Close']-df['Open'])/df['Open']*100
df['Daily_Return_%']=g['Close'].pct_change()*100
for w in [5,10,20,50]: df[f'MA_{w}']=g['Close'].transform(lambda s:s.rolling(w).mean())
df['EMA_12']=g['Close'].transform(lambda s:s.ewm(span=12,adjust=False).mean()); df['EMA_26']=g['Close'].transform(lambda s:s.ewm(span=26,adjust=False).mean())
for w in [10,20]: df[f'Volatility_{w}']=g['Daily_Return_%'].transform(lambda s:s.rolling(w).std())
for lag in [1,2,3,5,10]: df[f'Close_Lag_{lag}']=g['Close'].shift(lag); df[f'Return_Lag_{lag}']=g['Daily_Return_%'].shift(lag)
df['Volume_Change_%']=g['Volume'].pct_change()*100; df['Volume_MA_10']=g['Volume'].transform(lambda s:s.rolling(10).mean()); df['Volume_Ratio_10']=df['Volume']/df['Volume_MA_10']
df['Momentum_5']=g['Close'].transform(lambda s:s-s.shift(5)); df['Momentum_10']=g['Close'].transform(lambda s:s-s.shift(10))
df['Close_vs_MA_10_%']=(df['Close']-df['MA_10'])/df['MA_10']*100; df['Close_vs_MA_20_%']=(df['Close']-df['MA_20'])/df['MA_20']*100
df['Year']=df.Date.dt.year; df['Month']=df.Date.dt.month; df['Day']=df.Date.dt.day; df['Day_of_Week']=df.Date.dt.dayofweek
df['Future_Close']=g['Close'].shift(-1); df['Future_Return_%']=(df['Future_Close']-df['Close'])/df['Close']*100; df['Trend']=np.where(df['Future_Close']>df['Close'],1,0)
df=df.replace([np.inf,-np.inf],np.nan).dropna().reset_index(drop=True); df.to_csv(OUTPUT,index=False)
print('='*60); print('FEATURE ENGINEERING COMPLETED'); print('='*60); print(f'Rows: {len(df):,} | Columns: {len(df.columns)} | Stocks: {df.Ticker.nunique()}'); print(f'Saved: {OUTPUT}'); print(df.groupby('Ticker').size())
