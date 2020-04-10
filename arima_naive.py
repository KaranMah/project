#%matplotlib inline

import json
import math
import pprint
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import *
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error,r2_score, f1_score, precision_score, recall_score, roc_auc_score

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
# target = [metric]
features = ['Intraday_OC', 'Prev_close_open'] + [y+x for x in ['_Ret', '_MTD', '_YTD'] for y in metrics]# if (x+y) not in target]

def plot_results(y_true, y_pred, model):
    plot_df = pd.concat([y_true.reset_index(drop=True), pd.DataFrame(y_pred)], axis=1, ignore_index=True)
    plt.figure()
    plt.plot(plot_df)
    plt.title(model().__class__.__name__)

def run_auto_arima_model(train, test, features, target, is_exog=False):
    X_train, y_train = train
    X_test, y_test = test
    if(not is_exog):
        X_train = None
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
    print(model.summary())
    #model.plot_diagnostics(figsize=(7,5))
    #plt.show()
    n_periods = len(y_test)
    fc, confint = model.predict(n_periods=n_periods,
                                exogenous=X_test,
                                return_conf_int=True)
    index_of_fc = y_test.index
    fc_series = pd.Series(fc, index=index_of_fc)
    res['MSE'] = mean_squared_error(y_test, fc)
    res['R2'] = r2_score(y_test, fc)
    lower_series = pd.Series(confint[:, 0], index=index_of_fc)
    upper_series = pd.Series(confint[:, 1], index=index_of_fc)
    #plt.plot(y_train)
    #plt.plot(fc_series, color='darkgreen')
    #plt.plot(y_test, color='red')
    #plt.fill_between(lower_series.index,
    #                 lower_series,
    #                 upper_series,
    #                 color='k', alpha=.15)
    #plt.title("Final Forecast")
    #plt.show()
    # y_pred = reg.predict(X_test)
    # plot_results(pd.DataFrame(y_test), pd.DataFrame(y_pred), model)
    return(res, fc)


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
        try:
            scaler_y = scaler_y.fit(y_train.values.reshape(-1,1))
            y_train = scaler_y.transform(y_train.values.reshape(-1, 1))
        except:
            y_train = np.nan_to_num(y_train, posinf=1, neginf=-1)
            scaler_y = scaler_y.fit(y_train.reshape(-1,1))
            y_train = scaler_y.transform(y_train.reshape(-1, 1))
        try:
            y_test = scaler_y.transform(y_test.values.reshape(-1, 1))
        except:
            y_test = np.nan_to_num(y_test, posinf=1, neginf=-1)
            y_test = scaler_y.transform(y_test.reshape(-1, 1))
    return(X_train, X_test, y_train, y_test)

def check_bins(real, pred):
    y_test = Binarizer().transform(pd.DataFrame(real).pct_change().dropna())
    y_pred = Binarizer().transform(pd.DataFrame(pred).pct_change().dropna())
    return({"F1" :f1_score(y_test, y_pred, average='weighted'),
        "Precision": precision_score(y_test, y_pred, average='weighted'),
        "Recall": recall_score(y_test, y_pred, average='weighted')})

def do_forex_arima(cur, target, is_exog=False, transf = None):
    forex_cols = [x for x in forex.columns if x[1] == cur]
    X = forex[[col for col in forex_cols if col[0] in features + ['Time features']]][:-1]
    y = forex[[col for col in forex_cols if col[0] in target]].shift(-1)[:-1]
    X_train, X_test, y_train, y_test = split_scale(X, y, transf)
    res, y_pred = run_auto_arima_model((X_train, y_train), (X_test, y_test), features, target, is_exog)
    metrics = check_bins(y_test, y_pred)
    res.update(metrics)
    return(res)

def do_index_arima(cur, target, is_exog=False, transf = None):
    index_cols = [x for x in index.columns if x[1] == cur[0] and x[2] == cur[1]]
    X = index[[col for col in index_cols if col[0] in features + ['Time features']]][:-1]
    y = index[[col for col in index_cols if col[0] in target]].shift(-1)[:-1]
    X = X.dropna(how='any')
    y = y[y.index.isin(X.index)]
    X_train, X_test, y_train, y_test = split_scale(X, y, transf)
    res, y_pred = run_auto_arima_model((X_train, y_train), (X_test, y_test), features, target, is_exog)
    metrics = check_bins(y_test, y_pred)
    return(res.update(metrics))

def iterate_markets():
    reg_res = []
    for f_m in ['MNT', 'BDT', ('PKR', 'Karachi 100'), ('LKR', 'CSE All-Share')]:#(forex_pairs+index_pairs):
        print(f_m)
        for m in [metric, metric+'_Ret']:
            if(m == metric):
                temp_scalers = scalers[:1]
            else:
                temp_scalers = scalers
            for scaler in temp_scalers:
                for is_exog in [True, False]:
                    try:
                        if(f_m in forex_pairs):
                            res = do_forex_arima(f_m, [m], is_exog, scaler)
                        else:
                            res = do_index_arima(f_m, [m], is_exog, scaler)
                    except:
                        # res = do_forex_arima(f_m, [m], is_exog, scaler)
                        continue
                    res['Pair'] = f_m
                    res['Exog'] = is_exog
                    res['Target'] = m
                    res['Transformation'] = scaler().__class__.__name__ if scaler is not None else None
                    print(res)
                    reg_res.append(res)
    return(reg_res)

#
res = iterate_markets()
res_df = pd.DataFrame(res, columns= ['Pair', 'MSE', 'R2', 'Order', 'Seasonal_Order', 'AIC', 'BIC', 'F1', 'Precision', 'Recall'])
# # print(res_df)
res_df.to_csv("naive_arima.csv")
# res = do_forex_arima('MNT', 'Close', True, None)
# print(res)
