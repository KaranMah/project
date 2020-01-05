import json
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error,r2_score

def read_forex(filename="historical_forex.csv"):
    df = pd.read_csv(filename, parse_dates=["Date"])
    return df

def read_index(filename="historical_index.csv"):
    df = pd.read_csv(filename, parse_dates=["Date"])
    return df

forex = read_forex()
index = read_index()

us_dates = index[(index['Currency']=="USD") & (index['Idx']=="Nasdaq 100")]['Date']

pivoted_forex = pd.pivot_table(forex[forex['Date'].isin(us_dates)], index=["Date"], columns=["Currency"], values=["Open", "High", "Low", "Close"])
open_forex = pd.pivot_table(forex[forex['Date'].isin(us_dates)], index=["Date"], columns=["Currency"], values=["Open"])
close_forex = pd.pivot_table(forex[forex['Date'].isin(us_dates)], index=["Date"], columns=["Currency"], values=["Close"])

pivoted_index = pd.pivot_table(index[index['Date'].isin(us_dates)], index=["Date"], columns=["Currency", "Idx"], values=["Open", "High", "Low", "Close"])
open_index = pd.pivot_table(index[index['Date'].isin(us_dates)], index=["Date"], columns=["Currency", "Idx"], values=["Open"])
close_index = pd.pivot_table(index[index['Date'].isin(us_dates)], index=["Date"], columns=["Currency", "Idx"], values=["Close"])

open_forex.columns = map(lambda x: x[1], open_forex.columns)
open_forex = open_forex.drop('IDR', axis=1)

open_forex.isnull().values
open_forex[open_forex.isna().any(axis=1)]
len(open_forex)
open_forex.count()

plt.plot(open_forex.iloc[:,4])

ax = sns.heatmap(
    open_forex.corr(),
    vmin=-1, vmax=1, center=0,
    cmap=sns.diverging_palette(20, 220, n=200),
    square=True)

def run_sklearn_model(model, train, test, features, target):
    try:
        X_train, y_train = train
        X_test, y_test = test
    except:
        pass
    reg = model(max_iter=1000, tol=1e-3)
    reg.fit(X_train, y_train)
    y_pred = reg.predict(X_test)
    print("MSE: ", mean_squared_error(y_test, y_pred))
    print("R2: ", r2_score(y_test, y_pred))


def do_stuff_forex(raw_data, cur, periods):
    data = raw_data[cur]
    data = data.reset_index()
    data['ret'] = (data[[cur]] - data[[cur]].shift(1))/(data[[cur]].shift(1))
    for i in periods:
            data['ret_'+str(i)] = data['ret'].rolling(i).mean().shift()
    data = data.iloc[max(periods)+1:,:]
    print(data)
    plt.plot(data['ret'])
    plot_acf(data['ret'])
    plot_pacf(data['ret'])
    adf_res = adfuller(data['ret'])
    print("ADF statistic: ", adf_res[0])
    print("P - Value: ", adf_res[1])
    print("Lag: ", adf_res[2])
    features = list(map(lambda x: "ret_"+str(x), periods))
    X_train, X_test, y_train, y_test = train_test_split(data[features], data['ret'], test_size=0.33)
    run_sklearn_model(linear_model.SGDRegressor, (X_train, y_train), (X_test, y_test), features, 'ret')

do_stuff_forex(open_forex, "INR", [1, 3, 30])
