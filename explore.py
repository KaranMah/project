import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error,r2_score
from sklearn.ensemble import RandomForestRegressor

###############################################################################
###############################################################################

forex = pd.read_csv('prep_forex.csv', header=[0,1], index_col=0)
index = pd.read_csv('prep_index.csv', header=[0,1,2], index_col=0)


###############################################################################
###############################################################################

ax = sns.heatmap(
    forex['Close'].corr(),
    vmin=-1, vmax=1, center=0,
    cmap=sns.diverging_palette(20, 220, n=200),
    square=True)


###############################################################################
###############################################################################

def stationarity(forex, index, save=False):
    col_names = forex.columns
    forex_res = pd.DataFrame(index=pd.MultiIndex.from_tuples(col_names), columns=['ADF Statistic', 'P - Value', 'Lag'])
    for col, vals in forex.items():
        try:
            adf_res = adfuller(vals)
        except:
            adf_res = [np.nan, np.nan, np.nan]
        forex_res.loc[col, 'ADF Statistic'] = adf_res[0]
        forex_res.loc[col, 'P - Value'] = adf_res[1]
        forex_res.loc[col, 'Lag'] = adf_res[2]

    col_names = index.columns
    index_res = pd.DataFrame(index=pd.MultiIndex.from_tuples(col_names), columns=['ADF Statistic', 'P - Value', 'Lag'])
    for col, vals in index.items():
        try:
            adf_res = adfuller(vals)
        except:
            adf_res = [np.nan, np.nan, np.nan]
        index_res.loc[col, 'ADF Statistic'] = adf_res[0]
        index_res.loc[col, 'P - Value'] = adf_res[1]
        index_res.loc[col, 'Lag'] = adf_res[2]

    if(save):
        forex_res.to_csv('forex_stationarity.csv')
        index_res.to_csv('index_stationarity.csv')

# stationarity(forex, index, True)

def seasonal_decomposition(data, cur, met = ['Close_Ret']):
    col = tuple(met+cur)
    to_do = data.loc[:,col]
    result = seasonal_decompose(to_do, freq=250)
    result.plot()

seasonal_decomposition(forex, ['HKD'])


def plot_results(y_true, y_pred, model):
    plot_df = pd.concat([y_true.reset_index(drop=True), pd.Series(y_pred)], axis=1, ignore_index=True)
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


def do_stuff_forex(raw_data, cur, periods, model):
    data = raw_data[cur]
    data = data.reset_index()
    data['ret'] = (data[[cur]] - data[[cur]].shift(1))/(data[[cur]].shift(1))
    for i in periods:
            data['ret_'+str(i)] = data['ret'].rolling(i).mean().shift()
    data = data.iloc[max(periods)+1:,:]
    plt.plot(data['ret'])
    plot_acf(data['ret'])
    plot_pacf(data['ret'])
    features = list(map(lambda x: "ret_"+str(x), periods))
    X_train, X_test, y_train, y_test = train_test_split(data[features], data['ret'], test_size=0.33)
    run_sklearn_model(model, (X_train, y_train), (X_test, y_test), features, 'ret')

do_stuff_forex(open_forex, "INR", [1, 3, 30], RandomForestRegressor)
