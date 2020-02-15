import json
import pprint
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import *
from sklearn.model_selection import train_test_split
from sklearn.linear_model import *
from sklearn.metrics import mean_squared_error,r2_score

forex = pd.read_csv('prep_forex.csv', header=[0,1], index_col=0)
index = pd.read_csv('prep_index.csv', header=[0,1,2], index_col=0)

forex_pairs = list(set([x[1] for x in forex.columns if x[0] == 'Close']))
index_pairs = list(set([(x[1], x[2]) for x in index.columns if x[0] == 'Close']))

reg_models = [LinearRegression, Ridge, RidgeCV, Lasso, LassoCV, LassoLarsCV, LassoLarsIC, MultiTaskLasso,
          ElasticNet, ElasticNetCV, MultiTaskElasticNet, MultiTaskElasticNetCV, LassoLars,
          OrthogonalMatchingPursuit, BayesianRidge, ARDRegression,  LogisticRegression, LogisticRegressionCV,
          SGDRegressor, PassiveAggressiveRegressor, HuberRegressor, RANSACRegressor, TheilSenRegressor]

scalers = [None, MinMaxScaler, MaxAbsScaler, StandardScaler, RobustScaler, Normalizer, QuantileTransformer, PowerTransformer, FunctionTransformer]

metric = 'Close'
metrics = ['Open', 'Close', 'Low', 'High']
target = [metric+'_Ret']
features = ['Intraday_HL', 'Intraday_OC', 'Prev_close_open'] + [metric+x for x in ['_Ret', '_Ret_MA_3', '_Ret_MA_15', '_Ret_MA_45', '_MTD', '_YTD']]

def plot_results(y_true, y_pred, model):
    plot_df = pd.concat([y_true.reset_index(drop=True), pd.DataFrame(y_pred)], axis=1, ignore_index=True)
    plt.figure()
    plt.plot(plot_df)
    plt.title(model().__class__.__name__)

def run_sklearn_model(model, train, test, features, target):
    try:
        X_train, y_train = train
        X_test, y_test = test
    except:
        pass
    reg = model()#max_iter=1000, tol=1e-3)
    reg.fit(X_train, y_train)
    try:
        pprint.pprint(dict(zip(features,reg.feature_importances_)))
    except:
        pass
    y_pred = reg.predict(X_test)
    return({"MSE":mean_squared_error(y_test, y_pred),
            "R2" :r2_score(y_test, y_pred)})
    # print("MSE: ", mean_squared_error(y_test, y_pred))
    # print("R2: ", r2_score(y_test, y_pred))
    # plot_results(y_test, y_pred, model)

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


def do_forex(cur, model, transf = None, shuffle=False):
    forex_cols = [x for x in forex.columns if x[1] == cur]
    X = forex[[col for col in forex_cols if col[0] in features + ['Time features']]][:-1]
    y = forex[[col for col in forex_cols if col[0] in target]].shift(-1)[:-1]
    X = X.dropna(how='any')
    y = y[y.index.isin(X.index)]
    X_train, X_test, y_train, y_test = split_scale(X, y, transf, shuffle)
    res = run_sklearn_model(model, (X_train, y_train), (X_test, y_test), features, target)
    return(res)

def do_index(cur, model, transf = None, shuffle=False):
    index_cols = [x for x in index.columns if x[1] == cur[0] and x[2] == cur[1]]
    X = index[[col for col in index_cols if col[0] in features + ['Time features']]][:-1]
    y = index[[col for col in index_cols if col[0] in target]].shift(-1)[:-1]
    X = X.dropna(how='any')
    y = y[y.index.isin(X.index)]
    X_train, X_test, y_train, y_test = split_scale(X, y, transf, shuffle)
    res = run_sklearn_model(model, (X_train, y_train), (X_test, y_test), features, target)
    return(res)

def iterate_markets():
    reg_res = []
    for f_m in forex_pairs:
        print(f_m)
        for model in reg_models:
            for scaler in scalers:
                for shuffle in [True, False]:
                    try:
                        res = do_forex(f_m, model, scaler, shuffle)
                    except:
                        # res = do_index(f_m, model, scaler, shuffle)
                        continue
                    res['Pair'] = f_m
                    res['Transformation'] = scaler
                    res['Shuffle'] = shuffle
                    res['Model'] = model().__class__.__name__
                    reg_res.append(res)
    for f_m in index_pairs:
        print(f_m)
        for model in reg_models:
            for scaler in scalers:
                for shuffle in [True, False]:
                    try:
                        res = do_index(f_m, model, scaler, shuffle)
                    except:
                        #res = do_index(f_m, model, scaler, shuffle)
                        continue
                    res['Pair'] = f_m
                    res['Transformation'] = scaler
                    res['Shuffle'] = shuffle
                    res['Model'] = model().__class__.__name__
                    reg_res.append(res)
    return(reg_res)

res = iterate_markets()
res_df = pd.DataFrame(res, columns= ['Pair', 'Model', 'Transformation', 'Shuffle', 'MSE', 'R2'])
#print(res_df)
res_df.to_csv("linear_models_results.csv")
# do_stuff(["HKD", "Hang Seng"], LinearRegression)
