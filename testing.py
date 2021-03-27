import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import numpy as np 
from itertools import product 
import BacktesterSMA as BSMA
from scipy.optimize import brute
plt.style.use("seaborn")


tickers = ["XI", "AMZN", "MSFT", "XRP-USD", "BA", "KO", "IBM", "DIS", "MSFT" ]

def SMA(ticker):
    
    
    stocks = yf.download(ticker, start = "2008-01-01", end = "2021-03-26")
    
    df = stocks.loc[:, "Close"].copy()
    df = df.to_frame()
    print(df.columns)
    tester = BSMA.SMABacktester(df, ticker[0], 50, 200, "2004-01-01", "2021-06-30")
    
    
    
    tester.get_data()
    tester.set_parameters(50,200)
    
    result = tester.test_strategy()
    print(result)
    tester.plot_results()



SMA(["XI"])



