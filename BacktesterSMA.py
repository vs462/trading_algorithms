import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import brute
from itertools import product 
plt.style.use("seaborn")


class SMABacktester(): 
    def __init__(self, df, symbol, SMA_S, SMA_L, start, end):
        self.df = df.copy()
        self.symbol = symbol
        self.SMA_S = SMA_S        
        self.SMA_L = SMA_L
        self.start = start
        self.end = end
        self.get_data()
        
    def __repr__(self):
        return "SMABacktester(symbol = {}, SMA_S = {}, SMA_L = {}, start = {}, end = {})".format(self.symbol, self.SMA_S, self.SMA_L, self.start, self.end)
        
    def get_data(self):
        ''' Retrieves and prepares the data.
        '''
        raw = self.df
        raw = raw[self.symbol].to_frame().dropna()
        raw = raw.loc[self.start:self.end]
        raw.rename(columns={self.symbol: "price"}, inplace=True)
        raw["returns"] = np.log(raw / raw.shift(1))
        raw["SMA_S"] = raw["price"].rolling(self.SMA_S).mean()
        raw["SMA_L"] = raw["price"].rolling(self.SMA_L).mean()
        self.data = raw
        
    def set_parameters(self, SMA_S = None, SMA_L = None):
        ''' Updates SMA parameters and resp. time series.
        '''
        if SMA_S :
            self.SMA_S = SMA_S
            self.data["SMA_S"] = self.data["price"].rolling(self.SMA_S).mean()
        if SMA_L :
            self.SMA_L = SMA_L
            self.data["SMA_L"] = self.data["price"].rolling(self.SMA_L).mean()
            
    def test_strategy(self):
        ''' Backtests the trading strategy.
        '''
        data = self.data.copy().dropna()
        
        data["position"] = np.where(data["SMA_S"] > data["SMA_L"], 1, -1)
        data["strategy"] = data["position"].shift(1) * data["returns"]
        data.dropna(inplace=True)
        data["creturns"] = data["returns"].cumsum().apply(np.exp)
        data["cstrategy"] = data["strategy"].cumsum().apply(np.exp)
        self.results = data
        # absolute performance of the strategy
        perf = data["cstrategy"].iloc[-1]
        # out-/underperformance of strategy
        outperf = perf - data["creturns"].iloc[-1]
        
        return {'absolute performance':round(perf, 6), 'out-/underperformance of strategy':round(outperf, 6)}
    
    def plot_results(self):
        ''' Plots the cumulative performance of the trading strategy
        compared to buy and hold.
        '''
        if self.results is None:
            print("No results to plot yet. Run a strategy.")
        else:
            title = "{} | SMA_S = {} | SMA_L = {}".format(self.symbol, self.SMA_S, self.SMA_L)
            self.results[["creturns", "cstrategy"]].plot(title=title, figsize=(12, 8))
        
    def update_and_run(self, SMA):
        ''' Updates SMA parameters and returns the negative absolute performance (for minimization algorithm).

        Parameters
        ==========
        SMA: tuple
            SMA parameter tuple
        '''

        self.set_parameters(int(SMA[0]), int(SMA[1]))
        
        return -self.test_strategy()['absolute performance']

    def optimize_parameters(self, SMA_S_range, SMA_L_range):

        #print((range(SMA_S_range[0]), range(SMA_L_range)))
        combinations = list(product(range(SMA_S_range[0], SMA_S_range[1], SMA_S_range[2]), range(SMA_L_range[0], SMA_L_range[1], SMA_L_range[2]))) # create all possible combinations 
        print(combinations)
        results = []
        for comb in combinations:
            
            results.append(self.update_and_run(comb))
            
        best_comb = combinations[np.argmax(results)]
        
        return best_comb, np.max(results)

