import pandas as pd
INFILE='data/multi_stock_features.csv'; TRAIN='data/train.csv'; TEST='data/test.csv'
df=pd.read_csv(INFILE); df['Date']=pd.to_datetime(df['Date']); df=df.sort_values(['Ticker','Date'])
train=[]; test=[]
for _,g in df.groupby('Ticker'):
    i=int(len(g)*.8); train.append(g.iloc[:i]); test.append(g.iloc[i:])
pd.concat(train).to_csv(TRAIN,index=False); pd.concat(test).to_csv(TEST,index=False)
print(f'Train rows: {sum(map(len,train)):,}'); print(f'Test rows: {sum(map(len,test)):,}'); print('Saved data/train.csv and data/test.csv')
