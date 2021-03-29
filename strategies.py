import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import numpy as np 
from itertools import product 


ticker = ["SNOW", "AMZN", "MSFT"]#, "AMZN", "SNOW", "BA", "KO", "IBM", "DIS", "MSFT" ]

stocks = yf.download(ticker, start = "2008-01-01", end = "2021-03-26")

df = stocks.loc[:, "Close"].copy()

data = df['AMZN'].to_frame()
data.columns=['price']


def buy_and_hold(df):
    data = df.copy()
    data=data.to_frame()
    data.columns=['price']
    data["returns"] = np.log(data.price.div(data.price.shift(1)))
    data.dropna(inplace = True)
    return np.exp(data["returns"].sum())
    


def SMA_test(df, SMA_range, min_periods=10):
    data = df.copy()
    data.columns=['price']
    sma_s, sma_l = SMA_range[0], SMA_range[1]
    data["returns"] = np.log(data.price.div(data.price.shift(1)))
    data["SMA_S"] = data.price.rolling(int(sma_s), min_periods=min_periods).mean()
    data["SMA_L"] = data.price.rolling(int(sma_l), min_periods=min_periods).mean()
    data.dropna(inplace = True)
    
    data["position"] = np.where(data["SMA_S"] > data["SMA_L"], 1, -1)
    data["strategy"] = data.position.shift(1) * data["returns"]
    data.dropna(inplace = True)
    # risk
    # data[["returns", "strategy"]].std() * np.sqrt(252)
    return data[["returns", "strategy"]].sum().apply(np.exp)[-1] # maximize absolute performance


def plot_SMA(df, SMA_range, from_date='2000', min_periods=10):
    data = df.copy()
    sma_s, sma_l = SMA_range[0], SMA_range[1]
    data["SMA_S"] = data.price.rolling(int(sma_s), min_periods=min_periods).mean()
    data["SMA_L"] = data.price.rolling(int(sma_l), min_periods=min_periods).mean()
    data.dropna(inplace = True)
    
    data.loc[f'{from_date}':].plot(figsize = (12, 8), title = f"EUR/USD - SMA{sma_s} | SMA{sma_l}", fontsize = 12)
    plt.legend(fontsize = 12)
    plt.show()
 

def find_best_comb(df):
    data = df.copy()
    SMA_S_range = range(10,50,1)
    SMA_L_range = range(100, 252,1)
    
    combinations = list(product(SMA_S_range, SMA_L_range)) # create all possible combinations 
    
    results = []
    for comb in combinations:
        results.append(SMA_test(data, comb))
        
    best_comb = combinations[np.argmax(results)]
    
    return best_comb, np.max(results)

def cantrarian(df, window = 3):
    data = df.copy()
    # data=data.to_frame()
    data.columns=['price']
    data["returns"] = np.log(data.price.div(data.price.shift(1)))
    data.dropna(inplace = True)    
    data["position"] =  -np.sign(data["returns"].rolling(window).mean()) # contrarian (minus sign)
    data.dropna(inplace = True)
    data["strategy"] = data.position.shift(1) * data["returns"]
    return data[["returns", "strategy"]].sum().apply(np.exp)[-1]

def momentum(df, window = 3):
    data = df.copy()
    # data=data.to_frame()
    data.columns=['price']
    data["returns"] = np.log(data.price.div(data.price.shift(1)))
    data.dropna(inplace = True)
    data["position"] =  np.sign(data["returns"].rolling(window).mean()) # contrarian (minus sign)
    data.dropna(inplace = True)
    data["strategy"] = data.position.shift(1) * data["returns"]
    return data[["returns", "strategy"]].sum().apply(np.exp)[-1]

#data["position"] = np.where(data["SMA_S"] > data["SMA_L"], 1, -1 )

# result = SMA_test(data, (41, 100))
# result2 = buy_and_hold(data)


# print(result)
# plot_SMA(data, (50, 200), '2020-09')


# best = find_best_comb(data)

# SMA_S_range = range(10,50,1)
# SMA_L_range = range(100, 252,1)
# combinations = list(product(SMA_S_range, SMA_L_range))

test= data
bnh = cantrarian(test)
mom = momentum(test)
sma = SMA_test(test, (50,200))
print(bnh, mom, sma)































