import glob
import re
from backtesting import *
from backtesting import Backtest
from backtesting.lib import crossover
from backtesting.test import SMA
import pandas as pd
import numpy as np
from sklearn.svm import SVC

forex = pd.read_csv('prep_forex.csv', header=[0, 1], index_col=0)
index = pd.read_csv('prep_index.csv', header=[0, 1, 2], index_col=0)

forex_pairs = list(set([x[1] for x in forex.columns if x[0] == 'Close']))
index_pairs = list(set([(x[1], x[2]) for x in index.columns if x[0] == 'Close']))

cls_models = [SVC]

files = glob.glob("*_SVC_sliding*")


def majority(pred, n=4):
    res = pred.rolling(n).sum()
    return res


def getTom(pred):
    return pred


class ClasStrat(Strategy):
    window = 4
    n1 = 15
    n2 = 60

    def init(self):
        # Precompute two moving averages
        pred = pd.read_csv(cur + "_Close_prophet.csv")
        # res = iterate_markets(self.i)
        res = pred[['pred']]
        maj = np.array(majority(res, self.window)).ravel()
        self.sma1 = self.I(SMA, maj, self.n1)
        self.sma2 = self.I(SMA, maj, self.n2)
        self.tom = self.I(getTom, res)

    def next(self):
        if crossover(self.sma2, self.sma1) and self.tom[-1] == 1:
            self.sell()
        elif crossover(self.sma1, self.sma2) and self.tom[-1] == -1:
            self.buy()

def readData(curr):
    forex_features_bt = ["Open", "Close", "High", "Low", "Volume"]
    forex_cols_bt = [x for x in forex.columns if x[1] == curr]
    data = forex[[col for col in forex_cols_bt if col[0] in forex_features_bt]][:-1]
    data.columns = [x[0] for x in list(data.columns)]
    data = data.dropna(how='any')
    data.index = pd.to_datetime(data.index)
    return data


forex_pairs.sort()
for f_m in ["BDT"]:
    cur = f_m
    print(cur)
    data = readData(cur)
    data = data[int(len(data)*.8):]
    bt = Backtest(data, ClasStrat, cash=10000, commission=0.002)
    res = bt.run()
    print(res)
    bt.plot()
    # stats = bt.optimize(window=range(2, 10, 1),
    #                     i=range(lower_lim, upper_lim, 5),
    #                     n1=range(5, 30, 5),
    #                     n2=range(20, 80, 5),
    #                     maximize='Equity Final [$]',
    #                     constraint=lambda p: p.n1 < p.n2)
    # print(stats)
