import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import BacktesterSMA as BSMA
plt.style.use("seaborn")


tickers = ["XI", "AMZN", "MSFT", "XRP-USD", "BA", "KO", "IBM", "DIS", "MSFT" ]

def SMA(ticker):
    
    global df
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
stocks = yf.download(tickers, start = "2008-01-01", end = "2021-03-26")
    
df = stocks.loc[:, "Close"].copy()

print(df.columns)
tester = BSMA.SMABacktester(df, 'AMZN', 50, 200, "2004-01-01", "2021-06-30")
    
    
tester.get_data()
tester.set_parameters(50,200)
    
result = tester.test_strategy()
print(result)
tester.plot_results()
















