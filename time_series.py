%matplotlib inline

import json
import math
import pprint
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import *
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error,r2_score

import pmdarima as pm

forex = pd.read_csv('prep_forex.csv', header=[0,1], index_col=0)
index = pd.read_csv('prep_index.csv', header=[0,1,2], index_col=0)

forex_pairs = list(set([x[1] for x in forex.columns if x[0] == 'Close' and x[1] == 'INR']))
index_pairs = list(set([(x[1], x[2]) for x in index.columns if x[0] == 'Close' and x[1] == 'INR']))


scalers = [None, MinMaxScaler, MaxAbsScaler, StandardScaler, RobustScaler,
            PowerTransformer, FunctionTransformer]

metric = 'Close'
metrics = ['Open', 'Close', 'Low', 'High']
target = [metric]
features = ['Intraday_HL', 'Intraday_OC', 'Prev_close_open'] + [y+x for x in ['', '_Ret', '_Ret_MA_3', '_Ret_MA_15', '_Ret_MA_45', '_MTD', '_YTD'] for y in metrics]# if (x+y) not in target]

def plot_results(y_true, y_pred, model):
    plot_df = pd.concat([y_true.reset_index(drop=True), pd.DataFrame(y_pred)], axis=1, ignore_index=True)
    plt.figure()
    plt.plot(plot_df)
    plt.title(model().__class__.__name__)

def run_auto_arima_model(train, test, features, target):
    X_train, y_train = train
    X_test, y_test = test
    model = pm.auto_arima(y_train, start_p=1, start_q=1,
                      test='adf',       # use adftest to find optimal 'd'
                      max_p=3, max_q=3, # maximum p and q
                      m=5,              # frequency of series
                      d=None,           # let model determine 'd'
                      seasonal=True,    # No Seasonality
                      start_P=1,
                      D=1,
                      trace=True,
                      error_action='ignore',
                      suppress_warnings=True,
                      stepwise=True)

    print(model.summary())
    model.plot_diagnostics(figsize=(7,5))
    plt.show()
    n_periods = len(y_test)
    fc, confint = model.predict(n_periods=n_periods, return_conf_int=True)
    index_of_fc = X_test.index
    fc_series = pd.Series(fc, index=index_of_fc)
    lower_series = pd.Series(confint[:, 0], index=index_of_fc)
    upper_series = pd.Series(confint[:, 1], index=index_of_fc)
    plt.plot(y_train)
    plt.plot(fc_series, color='darkgreen')
    plt.plot(y_test, color='red')
    plt.fill_between(lower_series.index,
                     lower_series,
                     upper_series,
                     color='k', alpha=.15)
    plt.title("Final Forecast")
    plt.show()
    # y_pred = reg.predict(X_test)
    # plot_results(pd.DataFrame(y_test), pd.DataFrame(y_pred), model)
    # try:
    #     return({"MSE":mean_squared_error(y_test, y_pred),
    #         "R2" :r2_score(y_test, y_pred)})
    # except:
    #     y_pred = ([x[0] for x in y_pred])
    #     for i in range(len(y_pred)):
    #         if(np.isnan(y_pred[i])):
    #             y_pred[i] = 0
    #     return({"MSE":mean_squared_error(y_test, y_pred),
    #         "R2" :r2_score(y_test, y_pred)})

def split_scale(X, y, scaler, shuffle=False):
    X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=shuffle, test_size=0.2)
    if(scaler is None):
        return(X_train, X_test, y_train, y_test)
    else:
        scaler_X = scaler()
        if(scaler == scalers[-1]):
            scaler_X = scaler(np.log1p)
        scaler_X = scaler_X.fit(X_train)
        X_train = scaler_X.transform(X_train)
        X_test = scaler_X.transform(X_test)
        scaler_y = scaler()
        if(scaler == scalers[-1]):
            scaler_y = scaler(np.log1p)
        scaler_y = scaler_y.fit(y_train)
        y_train = scaler_y.transform(y_train)
        y_test = scaler_y.transform(y_test)
        return(X_train, X_test, y_train, y_test)

def do_forex(cur, transf = None, shuffle=False):
    forex_cols = [x for x in forex.columns if x[1] == cur]
    X = forex[[col for col in forex_cols if col[0] in features + ['Time features']]][:-1]
    y = forex[[col for col in forex_cols if col[0] in target]].shift(-1)[:-1]
    X_train, X_test, y_train, y_test = split_scale(X, y, transf, shuffle)
    res = run_auto_arima_model((X_train, y_train), (X_test, y_test), features, target)
    return(res)

def do_index(cur, transf = None, shuffle=False):
    index_cols = [x for x in index.columns if x[1] == cur[0] and x[2] == cur[1]]
    X = index[[col for col in index_cols if col[0] in features + ['Time features']]][:-1]
    y = index[[col for col in index_cols if col[0] in target]].shift(-1)[:-1]
    X = X.dropna(how='any')
    y = y[y.index.isin(X.index)]
    X_train, X_test, y_train, y_test = split_scale(X, y, transf, shuffle)
    res = run_auto_arima_model((X_train, y_train), (X_test, y_test), features, target)
    return(res)

do_index(index_pairs[1], None)
