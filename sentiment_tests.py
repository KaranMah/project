import json
import math
import pprint
import datetime
import numpy as np
import pandas as pd
from scipy.stats import zscore
import matplotlib.pyplot as plt

%matplotlib inline



forex = pd.read_csv('prep_forex.csv', header=[0,1], index_col=0)

def get_sentiment(cur):
    sent_map = {'BDT': 'Dhaka',
                     'MNT': 'UlaanBaatar',
                     ('PKR', 'Karachi 100'): 'Karachi',
                     ('LKR', 'CSE All-Share'): 'Colombo'}
    fill_vals = {'Avg': 0.5, 'Max': 0.5, 'Min': 0.5,
                 'Std': 0.0, 'Var': 0.0, 'Count': 0}
    f = lambda s: datetime.datetime.strptime(s,'%d-%m-%Y')
    ss = '#' + sent_map[cur]+ '_aggregated_data.csv'
    sd = pd.read_csv(ss, header=0, parse_dates=[0], date_parser=f,
                     names=['Date', 'Avg', 'Max', 'Min', 'Std', 'Var', 'Count'])
    sd = sd.set_index(sd['Date']).drop(['Date'], axis=1)
    sd.index = sd.index.strftime("%Y-%m-%d")
    sd = sd.fillna(value = fill_vals)
    col_names = (pd.MultiIndex.from_tuples([('Sentiment', x) for x in sd.columns]))
    sd.columns = col_names
    return(sd)


sent = get_sentiment('BDT')
market = forex[('Close_Ret', 'BDT')]
df = pd.concat([market, sent], axis=1)
print(sent.corrwith(market))
res = (sent[('Sentiment', 'Avg')])
# print(zscore(res))

plt.subplot(2, 1, 1)
plt.plot(market)
plt.title("BDT Close Returns")
plt.tick_params(
    axis='x',
    which='both',
    bottom=False,
    labelbottom=False)
plt.subplot(2, 1, 2)
plt.plot((res), color='orange')
plt.title("Bangladesh Sentiment Scores (Average)")
plt.tick_params(
    axis='x',
    which='both',
    bottom=False,
    labelbottom=False)
plt.show()
