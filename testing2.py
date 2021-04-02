import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import BacktesterSMA as BSMA
import strategies 
plt.style.use("seaborn")


tickers = ["XI", "AMZN", "MSFT", "XRP-USD", "BA", "KO", "IBM", "DIS", "MSFT" ]

def SMA(ticker):
    stocks = yf.download([ticker, 'XRP-USD'], start = "2008-01-01", end = "2021-03-26")
    
    df = stocks.loc[:, "Close"].copy()

    print(df.columns)
    tester = BSMA.SMABacktester(df, ticker, 50, 200, "2004-01-01", "2021-06-30")
    
    tester.get_data()
    tester.set_parameters(50,200)
    
    result = tester.test_strategy()
    print(result)
    tester.plot_results()



SMA("KO")

tickers = ["XI", "AMZN", "MSFT", "XRP-USD", "BA", "KO", "IBM", "DIS", "MSFT" ]
stocks = yf.download(tickers, start = "2008-01-01", end = "2021-03-30")
    
df = stocks.loc[:, "Close"].copy()

# test SMA strategy 
print(df.columns)
tester = BSMA.SMABacktester(df, 'KO', 50, 200, "2004-01-01", "2021-06-30")
    
tester.get_data()
tester.set_parameters(50,200)
    
result = tester.test_strategy()
print(result)
tester.plot_results()


# test all strategies 

tester = strategies.Strategies(df, 'KO', "2004-01-01", "2021-06-30")

print(tester.buy_and_hold())
print(tester.SMA(SMA_S =50, SMA_L=200))
print(tester.momentum(window = 3))
print(tester.cantrarian(window = 3))
      


def plot_SMA(df, SMA_range=(50,200), from_date='2000', min_periods=10):
    data = df.copy()
    data = data[['BA']]
    sma_s, sma_l = SMA_range[0], SMA_range[1]
    data.columns = ["price"]
    print(data)

    data["SMA_S"] = data.price.rolling(int(sma_s), min_periods=min_periods).mean()
    data["SMA_L"] = data.price.rolling(int(sma_l), min_periods=min_periods).mean()
    data.dropna(inplace = True)  
    data.loc[f'{from_date}':].plot(figsize = (12, 8), title = f"EUR/USD - SMA{sma_s} | SMA{sma_l}", fontsize = 12)
    plt.legend(fontsize = 12)
    plt.show()


x = plot_SMA(df)











