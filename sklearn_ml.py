import pprint
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error,r2_score

forex = pd.read_csv('prep_forex.csv', header=[0,1], index_col=0)
index = pd.read_csv('prep_index.csv', header=[0,1,2], index_col=0)

metric = 'Close'
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
    print("MSE: ", mean_squared_error(y_test, y_pred))
    print("R2: ", r2_score(y_test, y_pred))
    plot_results(y_test, y_pred, model)


def do_stuff(cur, model):
    forex_cols = [x for x in forex.columns if x[1] == cur[0]]
    index_cols = [x for x in index.columns if x[1] == cur[0] and x[2] == cur[1]]
    X = forex[[col for col in forex_cols if col[0] in features + ['Time features']]][:-1]
    y = forex[[col for col in forex_cols if col[0] in target]].shift(-1)[:-1]
    X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=False, test_size=0.2)
    run_sklearn_model(model, (X_train, y_train), (X_test, y_test), features, 'ret')

do_stuff(["HKD", "Hang Seng"], linear_model.LinearRegression)
