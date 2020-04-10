from backtesting import *
from backtesting import Backtest
from backtesting.lib import crossover
from backtesting.test import SMA
import pandas as pd
from backtesting.test import GOOG


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


def readData (curr):
    forex_features = ["Open", "Close", "High", "Low", "Volume"]
    forex = pd.read_csv('prep_forex.csv', header=[0, 1], index_col=0)
    forex_cols = [x for x in forex.columns if x[1] == curr]
    data = forex[[col for col in forex_cols if col[0] in forex_features]][:-1]
    data.columns = [x[0] for x in list(data.columns)]
    data.index = pd.to_datetime(data.index)
    print(data.head())
    return data

strat = ["Classification", "Regressiont"] 

data = readData("INR")
bt = Backtest(data, ClasStrat, cash=10000, commission=0.0001)
res = bt.run()
print(res)
bt.plot()