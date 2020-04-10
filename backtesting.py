from backtesting import Strategy
from backtesting.lib import crossover
from backtesting import Backtest
import pandas as pd


class ClasStrat(Strategy):
    
    # Define the two MA lags as *class variables*
    # for later optimization
    n1 = 10
    n2 = 20
    
    def init(self):
        # Precompute two moving averages
        self.sma1 = self.I(SMA, self.data.Close, self.n1)
        self.sma2 = self.I(SMA, self.data.Close, self.n2)
    
    def next(self):
        # If sma1 crosses above sma2, buy the asset
        if crossover(self.sma1, self.sma2):
            self.buy()

        # Else, if sma1 crosses below sma2, sell it
        elif crossover(self.sma2, self.sma1):
            self.sell()

def readData (start):
    data = pd.read_csv(start+"results.csv").to_datetime() #confirm
    return data

strat = ["Classification", "Regressiont"] 

data = readData(strat)
bt = backtest(data, ClasStrat, cash = 10000, commision = 0.02)
res = bt.run()
print(res)
bt.plot()