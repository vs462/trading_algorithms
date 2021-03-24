import pandas as pd
import tpqoa # wrapper 
import yfinance as yf
import matplotlib.pyplot as plt
import numpy as np 


ticker = ["SNOW", "AMZN", "MSFT"]#, "AMZN", "SNOW", "BA", "KO", "IBM", "DIS", "MSFT" ]

stocks = yf.download(ticker, start = "2008-01-01", end = "2021-03-26")

df = stocks.loc[:, "Open"].copy()

data = df['SNOW'].to_frame()
data.columns=['price']

 
def run_strategy(df, SMA):
    data = df.copy()
    data["returns"] = np.log(data.price.div(data.price.shift(1)))
    data["SMA_S"] = data.price.rolling(int(SMA[0]), min_periods=10).mean()
    data["SMA_L"] = data.price.rolling(int(SMA[1]), min_periods=10).mean()
    data.dropna(inplace = True)
    
    data["position"] = np.where(data["SMA_S"] > data["SMA_L"], 1, -1)
    data["strategy"] = data.position.shift(1) * data["returns"]
    data.dropna(inplace = True)
    
    
    return data[["returns", "strategy"]].sum().apply(np.exp)#[-1] # maximize absolute performance

result = run_strategy(data, (50, 200))
print(result)

sma_s = 50
sma_l = 200

data["SMA_S"] = data.price.rolling(sma_s, min_periods=10).mean()
data["SMA_L"] = data.price.rolling(sma_l).mean()


data.plot(figsize = (12, 8), title = f"EUR/USD - SMA{sma_s} | SMA{sma_l}", fontsize = 12)
plt.legend(fontsize = 12)
plt.show()


data.loc["2020-09":"2021"].plot(figsize = (12, 8), title = f"EUR/USD - SMA{sma_s} | SMA{sma_l}", fontsize = 12)
plt.legend(fontsize = 12)
plt.show()

data["position"] = np.where(data["SMA_S"] > data["SMA_L"], 1, -1 )
