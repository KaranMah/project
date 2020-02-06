import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import seasonal_decompose


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
    plot_acf(to_do)
    plot_pacf(to_do)

seasonal_decomposition(forex, ['HKD'])
