import os, numpy as np, pandas as pd, matplotlib.pyplot as plt
INPUT='data/backtest_prices.csv'; os.makedirs('results',exist_ok=True)
INITIAL=100000.0; COST=0.001
x=pd.read_csv(INPUT); x['Date']=pd.to_datetime(x['Date']); x=x.sort_values('Date').reset_index(drop=True)
x['MA_20']=x['Close'].rolling(20).mean(); x['MA_50']=x['Close'].rolling(50).mean(); x['Signal']=np.where(x['MA_20']>x['MA_50'],1,0); x.loc[x['MA_50'].isna(),'Signal']=0
x['Position']=x['Signal'].shift(1).fillna(0); x['Market_Return']=x['Close'].pct_change().fillna(0); x['Buy_Hold_Return']=x['Market_Return']; x['Trade']=x['Position'].diff().abs().fillna(0); x['Transaction_Cost']=x['Trade']*COST
x['Strategy_Return']=x['Position']*x['Market_Return']-x['Transaction_Cost']; x['Strategy_Equity']=INITIAL*(1+x['Strategy_Return']).cumprod(); x['Buy_Hold_Equity']=INITIAL*(1+x['Buy_Hold_Return']).cumprod()
for c in ['Strategy','Buy_Hold']:
    eq=x[f'{c}_Equity']; x[f'{c}_Drawdown']=eq/eq.cummax()-1

def metrics(r,eq):
    total=eq.iloc[-1]/INITIAL-1; n=len(r); ann=(1+total)**(252/n)-1 if n else 0; vol=r.std()*np.sqrt(252); sharpe=r.mean()/r.std()*np.sqrt(252) if r.std()!=0 else 0; dd=(eq/eq.cummax()-1).min(); active=(r!=0).sum(); win=(r>0).sum()/active if active else 0
    return [INITIAL,eq.iloc[-1],total*100,ann*100,vol*100,sharpe,dd*100,win*100]
summary=pd.DataFrame([metrics(x['Strategy_Return'],x['Strategy_Equity']),metrics(x['Buy_Hold_Return'],x['Buy_Hold_Equity'])],index=['Moving Average Strategy','Buy & Hold'],columns=['Initial Capital','Final Capital','Total Return %','Annualized Return %','Annualized Volatility %','Sharpe Ratio','Maximum Drawdown %','Winning Days %'])
x.to_csv('results/backtest_results.csv',index=False); summary.to_csv('results/backtest_summary.csv')
plt.figure(figsize=(12,6)); plt.plot(x.Date,x.Strategy_Equity,label='Moving Average Strategy'); plt.plot(x.Date,x.Buy_Hold_Equity,label='Buy & Hold'); plt.title('Backtest: Strategy vs Historical Buy & Hold'); plt.xlabel('Date'); plt.ylabel('Portfolio Value'); plt.legend(); plt.grid(True); plt.tight_layout(); plt.savefig('results/equity_curve.png',dpi=150); plt.close()
plt.figure(figsize=(12,5)); plt.plot(x.Date,x.Strategy_Drawdown*100,label='Strategy Drawdown'); plt.plot(x.Date,x.Buy_Hold_Drawdown*100,label='Buy & Hold Drawdown'); plt.title('Historical Drawdown'); plt.xlabel('Date'); plt.ylabel('Drawdown (%)'); plt.legend(); plt.grid(True); plt.tight_layout(); plt.savefig('results/drawdown.png',dpi=150); plt.close()
print(summary.round(2)); print('Trades:',int(x.Trade.sum())); print('Saved results in results/')
