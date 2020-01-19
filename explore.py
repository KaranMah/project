import json
import pprint
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
from sklearn.ensemble import RandomForestRegressor

def interpolate_data(save=False):
    forex = pd.read_csv("historical_forex.csv", parse_dates=["Date"])
    pivoted_forex = pd.pivot_table(forex, index=["Date"], columns=["Currency"], values=["Open", "High", "Low", "Close"])
    first_date = min(pivoted_forex.index)
    last_date = max(pivoted_forex.index)
    temp_dates = list(filter(lambda x: x.weekday() not in [5,6], pd.date_range(first_date, last_date)))
    pivoted_forex = pivoted_forex.reindex(temp_dates)
    for i in range(len(pivoted_forex.columns)):
        pivoted_forex.iloc[:,i] = pivoted_forex.iloc[:,i].interpolate(method="time")
    if(save):
        pivoted_forex.to_csv('interpolated_historical_forex.csv')

    index = pd.read_csv("historical_index.csv", parse_dates=["Date"])
    pivoted_index = pd.pivot_table(index, index=["Date"], columns=["Currency", "Idx"], values=["Open", "High", "Low", "Close", "Volume"])
    first_date = min(pivoted_index.index)
    last_date = max(pivoted_index.index)
    temp_dates = list(filter(lambda x: x.weekday() not in [5,6], pd.date_range(first_date, last_date)))
    pivoted_index = pivoted_index.reindex(temp_dates)
    for i in range(len(pivoted_index.columns)):
        pivoted_index.iloc[:,i] = pivoted_index.iloc[:,i].interpolate(method="time")
    if(save):
        pivoted_index.to_csv('interpolated_historical_index.csv')

    return (pivoted_forex, pivoted_index)

def oc_return(data, index = True):
    intraday = ((data['Close']-data['Open'])/data['Open'])
    if(index):
        intraday.columns = pd.MultiIndex.from_tuples([('Intraday',)+x for x in intraday.columns])
    else:
        intraday.columns = pd.MultiIndex.from_tuples([('Intraday',x) for x in intraday.columns])
    concat_data = data.join(intraday)
    return(concat_data)

def get_returns(data, index=True):
    concat_data = data
    raws = ['Open', 'High', 'Low', 'Close', 'Volume']
    for met in raws:
        try:
            temp = data[met].pct_change(1)
        except:
            continue
        if(index):
            temp.columns = pd.MultiIndex.from_tuples([(met+'_Ret',)+x for x in temp.columns])
        else:
            temp.columns = pd.MultiIndex.from_tuples([(met+'_Ret',x) for x in temp.columns])
        concat_data = concat_data.join(temp)
    return(concat_data)

def add_features(forex= None, index= None, save=False):
    forex = (pd.read_csv("interpolated_historical_forex.csv", header=[0,1], index_col=0)) if forex is None else forex
    transf_forex = oc_return(forex, False)
    transf_forex = get_returns(transf_forex, False)
    if(save):
        transf_forex.to_csv('extra_features_forex.csv')

    index = pd.read_csv("interpolated_historical_index.csv", header=[0,1,2], index_col=0) if index is None else index
    transf_index = oc_return(index, True)
    transf_index = get_returns(transf_index, True)
    if(save):
        transf_index.to_csv('extra_features_index.csv')

    return(transf_forex, transf_index)

def transform(data, start_date = '01-01-2010', end_date = '12-31-2019'):
    dates = list(filter(lambda x: x.weekday() not in [5,6], pd.date_range(start_date, end_date)))
    return(data[data.index.isin(dates)])


print("Interpolating forex and index rates...")
try:
    interpolated_forex, interpolated_index = interpolate_data(True)
    print("Done interpolating...")
except:
    print("Some error happened in interpolation...")
print("Adding extra features...")
try:
    updated_forex, updated_index = add_features(interpolated_forex, interpolated_index)
    print("Done updating features...")
except:
    print("Some error in adding features...")
print("Transforming dates...")
try:
    prep_forex = transform(updated_forex)
    prep_index = transform(updated_index)
    print("Done transforming for dates...")
except:
    print("Some error in transforming dates...")


###############################################################################
###############################################################################

ax = sns.heatmap(
    open_forex.corr(),
    vmin=-1, vmax=1, center=0,
    cmap=sns.diverging_palette(20, 220, n=200),
    square=True)

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
    adf_res = adfuller(data['ret'])
    print("ADF statistic: ", adf_res[0])
    print("P - Value: ", adf_res[1])
    print("Lag: ", adf_res[2])
    features = list(map(lambda x: "ret_"+str(x), periods))
    X_train, X_test, y_train, y_test = train_test_split(data[features], data['ret'], test_size=0.33)
    run_sklearn_model(model, (X_train, y_train), (X_test, y_test), features, 'ret')

do_stuff_forex(open_forex, "INR", [1, 3, 30], RandomForestRegressor)
