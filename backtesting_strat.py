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

class ClasStrat(Strategy):
    window = 8
    i = 30
    n1 = 15
    n2 = 60

    def init(self):
        # Precompute two moving averages
        pred = pd.read_csv(cur + "_SVC_sliding_"+str(self.i)+"_results.csv")
        #res = iterate_markets(self.i)
        res = pred[['pred']]
        maj = np.array(majority(res, self.window)).ravel()
        self.sma1 = self.I(SMA, maj, self.n1)
        self.sma2 = self.I(SMA, maj, self.n2)
    
    def next(self):
        if crossover(self.sma1, self.sma2):
            self.buy()

            # Else, if sma1 crosses below sma2, sell it
        elif crossover(self.sma2, self.sma1):
            self.sell()


def readData (curr):
    forex_features_bt = ["Open", "Close", "High", "Low", "Volume"]
    forex_cols_bt = [x for x in forex.columns if x[1] == curr]
    data = forex[[col for col in forex_cols_bt if col[0] in forex_features_bt]][:-1]
    data.columns = [x[0] for x in list(data.columns)]
    data.index = pd.to_datetime(data.index)
    return data

forex_pairs.sort()
for f_m in forex_pairs:
    cur_files = [f for f in files if f_m in f]
    cur_files.sort()
    if len(cur_files) != 0:
        lower_lim = int(re.findall('\d+', cur_files[0])[0])
        upper_lim = int(re.findall('\d+', cur_files[len(cur_files) - 1])[0]) + 5
        cur = f_m
        print(cur)
        data = readData(cur)
        bt = Backtest(data, ClasStrat, cash=10000, commission=0.002)
        stats = bt.optimize(window=range(2, 10, 1),
                            i=range(lower_lim, upper_lim, 5),
                            n1=range(5, 30, 5),
                            n2=range(20, 80, 5),
                            maximize='Equity Final [$]',
                            constraint=lambda p: p.n1 < p.n2)
        print(stats)

# metric = 'Close'
# metrics = ['Open', 'Close', 'Low', 'High']
# target = [metric + '_Ret']
#
# forex_features = ['Intraday_HL', 'Intraday_OC', 'Prev_close_open'] + [y + x for x in
#                                                                       ['_Ret', '_Ret_MA_3', '_Ret_MA_15', '_Ret_MA_45',
#                                                                        '_MTD', '_YTD'] for y in metrics]
# index_features = ['Intraday_HL', 'Intraday_OC', 'Prev_close_open'] + [y + x for x in
#                                                                       ['_Ret', '_Ret_MA_3', '_Ret_MA_15', '_Ret_MA_45',
#                                                                        '_MTD', '_YTD'] for y in (metrics + ['Volume'])]
#
# scalers = [Binarizer]
#
#
# def run_sklearn_model(model, train, test, features, target):
#     X_train, y_train = train
#     X_test, y_test = test
#
#     period = len(X_train)
#     y_train, y_test = y_train.astype(int), y_test.astype(int)
#     # y_train = y_train.values
#     # y_test = y_test.values
#     prediction = []
#     data = X_train.values
#     data_y = y_train
#     for i, t in enumerate(X_test.values):
#         reg = model()
#         reg.fit(data, data_y)
#         if i == 0:
#             y = reg.predict(X_train)
#             for elem in y:
#                 prediction.append(elem)
#         y = reg.predict([t])
#         prediction.append(y[0])
#         data = np.vstack((data, t))
#         data = np.delete(data, 0, 0)
#         data_y = np.append(data_y, y_test[i])
#         data_y = np.delete(data_y, 0, 0)
#
#     y_true = np.vstack((y_train,y_test))
#     print(model.__name__ + " " + str(period) + " accuracy = ", accuracy_score(y_true, prediction))
#     # print(y_test, prediction)
#     print(confusion_matrix(y_true, prediction))
#     pred = pd.DataFrame(prediction)
#     return pred
#
#
# def split_scale(X, y, scaler, train_index, test_index, shuffle=False, poly=False, transf_features_also=False):
#     X_train, X_test = X.iloc[:train_index], X.iloc[train_index:]
#     y_train, y_test = y.iloc[:train_index], y.iloc[train_index:]
#     if scaler:
#         scaler_X = scaler()
#         if (transf_features_also):
#             X_train = scaler_X.transform(X_train)
#             X_test = scaler_X.transform(X_test)
#         scaler_y = scaler()
#
#         y_train = np.array(y_train).reshape(1, -1)
#         y_test = np.array(y_test).reshape(1, -1)
#         y_train = scaler_y.transform(y_train)
#         y_test = scaler_y.transform(y_test)
#         y_train[y_train == 0] = -1
#         y_test[y_test == 0] = -1
#         y_test = np.array(y_test).reshape(-1, 1)
#         y_train = np.array(y_train).reshape(-1, 1)
#
#     return (X_train, X_test, y_train, y_test)
#
#
# def do_forex(cur, model, train_index, test_index, transf=None, shuffle=False, poly=False, transf_features_also=False):
#     forex_cols = [x for x in forex.columns if x[1] == cur]
#     X = forex[[col for col in forex_cols if col[0] in forex_features + ['Time features']]][:-1]
#     y = forex[[col for col in forex_cols if col[0] in target]].shift(-1)[:-1]
#     X = X.dropna(how='any')
#     y = y[y.index.isin(X.index)]
#     X_train, X_test, y_train, y_test = split_scale(X, y, transf, train_index, test_index, shuffle, poly,
#                                                    transf_features_also)
#     # res = walk_forward((X_train, y_train), (X_test, y_test))
#     res = run_sklearn_model(model, (X_train, y_train), (X_test, y_test), forex_features, target)
#     return (res)
#
#
# def do_index(cur, model, train_index, test_index, transf=None, shuffle=False, poly=False, transf_features_also=False):
#     index_cols = [x for x in index.columns if x[1] == cur[0] and x[2] == cur[1]]
#     X = index[[col for col in index_cols if col[0] in index_features + ['Time features']]][:-1]
#     y = index[[col for col in index_cols if col[0] in target]].shift(-1)[:-1]
#     X = X.dropna(how='any')
#     y = y[y.index.isin(X.index)]
#     X_train, X_test, y_train, y_test = split_scale(X, y, transf, train_index, test_index, transf, shuffle, poly,
#                                                    transf_features_also)
#     res = run_sklearn_model(model, (X_train, y_train), (X_test, y_test), index_features, target)
#     # res = run_sklearn_model((X_train, y_train), (X_test, y_test))
#     return (res)
#
# def iterate_markets(i):
#     shuffle = False
#     poly = False
#     transf_features_also = False
#     f_m = cur
#     model = SVC
#     scaler = Binarizer
#     reg_res = pd.DataFrame()
#     if (f_m in forex_pairs):
#         train_index = i
#         test_index = 0
#         res = do_forex(f_m, model, train_index, test_index, scaler, shuffle, poly,
#                        transf_features_also)
#         reg_res = res
#     else:
#         train_index = i
#         test_index = 0
#
#         res = do_index(f_m, model, train_index, test_index, scaler, shuffle, poly,
#                        transf_features_also)
#         reg_res = pd.concat([reg_res, res])
#         reg_res.columns = ["pred"]
#     return reg_res