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

cur = ('INR', 'Nifty 50')
# cur = 'INR'

forex_pairs = list(set([x[1] for x in forex.columns if x[0] == 'Close']))
index_pairs = list(set([(x[1], x[2]) for x in index.columns if x[0] == 'Close']))

scalers = [None, MinMaxScaler, MaxAbsScaler, StandardScaler, RobustScaler, Normalizer,
           QuantileTransformer, PowerTransformer, FunctionTransformer]

metric = 'Close'
metrics = ['Open', 'Close', 'Low', 'High', 'Volume']
target = [metric]
features = ['Intraday_OC', 'Prev_close_open'] + [y+x for x in ['_Ret', '_MTD', '_YTD'] for y in metrics]# if (x+y) not in target]

def plot_results(y_true, y_pred, model):
    plot_df = pd.concat([y_true.reset_index(drop=True), pd.DataFrame(y_pred)], axis=1, ignore_index=True)
    plt.figure()
    plt.plot(plot_df)
    plt.title(model().__class__.__name__)

def run_auto_arima_model(train, test, features, target):
    X_train, y_train = train
    X_test, y_test = test
    model = pm.auto_arima(y_train, start_p=1, start_q=1,
                      exogenous=X_train,
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
    res = {}
    res['Order'] = model.order
    res['Seasonal_Order'] = model.seasonal_order
    res['AIC'] = model.aicc()
    res['BIC'] = model.bic()
    # print(model.summary())
    # model.plot_diagnostics(figsize=(7,5))
    # plt.show()
    n_periods = len(y_test)
    fc, confint = model.predict(n_periods=n_periods,
                                exogenous=X_test,
                                return_conf_int=True)
    index_of_fc = y_test.index
    fc_series = pd.Series(fc, index=index_of_fc)
    res['MSE'] = mean_squared_error(y_test, fc)
    res['R2'] = r2_score(y_test, fc)
    # lower_series = pd.Series(confint[:, 0], index=index_of_fc)
    # upper_series = pd.Series(confint[:, 1], index=index_of_fc)
    # plt.plot(y_train)
    # plt.plot(fc_series, color='darkgreen')
    # plt.plot(y_test, color='red')
    # plt.fill_between(lower_series.index,
    #                  lower_series,
    #                  upper_series,
    #                  color='k', alpha=.15)
    # plt.title("Final Forecast")
    # plt.show()
    # y_pred = reg.predict(X_test)
    # plot_results(pd.DataFrame(y_test), pd.DataFrame(y_pred), model)
    return(res)


def split_scale(X, y, scaler):
    X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=False, test_size=0.2)
    if(scaler):
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
    X_train = X_train.replace({np.inf: 1, -np.inf: -1})
    y_train = y_train.replace({np.inf: 1, -np.inf: -1})
    X_test = X_test.replace({np.inf: 1, -np.inf: -1})
    y_test = y_test.replace({np.inf: 1, -np.inf: -1})
    return(X_train, X_test, y_train, y_test)

def do_forex_index(cur, target, transf=None):
    index_cols = [x for x in index.columns if x[1] == cur]
    forex_cols = [x for x in forex.columns if x[1] == cur]
    index_X = index[[col for col in index_cols if col[0] in features + ['Time features']]][:-1]
    index_y = index[[col for col in index_cols if col[0] in target]].shift(-1)[:-1]
    forex_X = forex[[col for col in forex_cols if col[0] in features + ['Time features']]][:-1]
    forex_y = forex[[col for col in forex_cols if col[0] in target]].shift(-1)[:-1]
    index_X.columns = index_X.columns.values
    forex_X.columns = forex_X.columns.values
    X = pd.concat([index_X, forex_X], axis=1)
    y = pd.concat([index_y, forex_y], axis=1)
    y_cols = np.concatenate((index_y.columns.values, forex_y.columns.values))
    X = X.dropna(how='all', axis=1)
    X = X.dropna(how='any')
    y = y[y.index.isin(X.index)]
    tot_res = []
    for y_col in (y_cols):
        y_temp = y[y_col]
        res = []
        X_train, X_test, y_train, y_test = split_scale(X, y_temp, transf)
        try:
            res = run_auto_arima_model((X_train, y_train), (X_test, y_test), features, target)
            res['Pair'] = y_col
            print(res)
            tot_res.append(res)
        except:
            continue
    return(tot_res)

def iterate_markets():
    reg_res = []
    for f_m in forex_pairs[:1]:
        print(f_m)
        for m in [metric, metric+'_Ret']:
            if(m == metric):
                temp_scalers = scalers[:1]
            else:
                temp_scalers = scalers
            for scaler in temp_scalers:
                res = do_forex_index(f_m, [m], scaler)
                print(res)
                reg_res.extend(res)
    return(reg_res)


res = iterate_markets()
res_df = pd.DataFrame(res, columns= ['Pair', 'MSE', 'R2', 'Order', 'Seasonal_Order', 'AIC', 'BIC'])
# print(res_df)
res_df.to_csv("arima_forex_index.csv")
