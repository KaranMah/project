from backtesting import *
from backtesting import Backtest
from backtesting.lib import crossover, plot_heatmaps
from backtesting.test import SMA
import pandas as pd
import numpy as np
import seaborn

forex = pd.read_csv('prep_forex.csv', header=[0, 1], index_col=0)
index = pd.read_csv('prep_index.csv', header=[0, 1, 2], index_col=0)

forex_pairs = list(set([x[1] for x in forex.columns if x[0] == 'Close']))
index_pairs = list(set([(x[1], x[2]) for x in index.columns if x[0] == 'Close']))

csv_dir = "./walk_forward_opt/final/"
xstr = lambda s: '' if s is None else str(s)

def majority(pred, n=4):
    res = pred.rolling(n).sum()
    return res


def getTom(pred):
    return pred


class ClasStrat(Strategy):
    window = 2
    n1 = 15
    n2 = 45

    def init(self):
        pred = pd.read_csv(csv_dir + str(cur) + "_final" + xstr(win) + ".csv")
        res = pred[['pred_mode']]
        res = res[int(len(res)*.8):]
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
    data = pd.DataFrame()
    if not isinstance(curr, tuple):
        csv = "prep_forex.csv"
        data = pd.read_csv(csv, header=[0, 1], index_col=0)
        forex_features_bt = ["Open", "Close", "High", "Low", "Volume"]
        forex_cols_bt = [x for x in data.columns if x[1] == curr]
        data = data[[col for col in forex_cols_bt if col[0] in forex_features_bt]][:-1]
        data.columns = [x[0] for x in list(data.columns)]
        data = data.dropna(how='any')

    else:
        csv = "prep_index.csv"
        data = pd.read_csv(csv, header=[0, 1, 2], index_col=0)
        index_features_bt = ["Open", "Close", "High", "Low", "Volume"]
        index_cols_bt = [x for x in data.columns if x[1] == curr[0] and x[2] == curr[1]]
        data = data[[col for col in index_cols_bt if col[0] in index_features_bt]][:-1]
        data.columns = [x[0] for x in list(data.columns)]
        data = data.dropna(how='any')
    data.index = pd.to_datetime(data.index)

    return data

target_markets = ['BDT', 'MNT', ('PKR', 'Karachi 100'), ('LKR', 'CSE All-Share')]
window = {"MNT": [None, "_40"],
          ('PKR', 'Karachi 100'): [None, "_45"],
          ('LKR', 'CSE All-Share'): [None, "_30"],
          "BDT": [None, "_50"]}

forex_pairs.sort()
for f_m in target_markets:
    cur = f_m
    print(cur)
    for win in window[f_m]:
        data = readData(cur)
        data = data[int(len(data)*.8):]
        bt = Backtest(data, ClasStrat, cash=100000, commission=0.002, trade_on_close=True)
        res = bt.run()
        print(res)
        fileName = "./images/bt_plot/opt_"+str(cur) + xstr(win)
        bt.plot(filename=fileName)
        stats = bt.optimize(window=range(2, 10, 1),
                             i=range(lower_lim, upper_lim, 5),
                             n1=range(5, 30, 5),
                             n2=range(20, 80, 5),
                             maximize='Equity Final [$]',
                             constraint=lambda p: p.n1 < p.n2)
        print(stats)
        
