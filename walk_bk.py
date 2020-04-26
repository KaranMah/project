import datetime
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import *
from sklearn.metrics import *
from sklearn.preprocessing import *
from sklearn.svm import *
import warnings
from sklearn.exceptions import DataConversionWarning
import matplotlib.pyplot as plt

warnings.filterwarnings(action='ignore', category=DataConversionWarning)
warnings.filterwarnings(action='ignore', category=UserWarning)

forex = pd.read_csv('prep_forex.csv', header=[0, 1], index_col=0)
index = pd.read_csv('prep_index.csv', header=[0, 1, 2], index_col=0)

forex_pairs = list(set([x[1] for x in forex.columns if x[0] == 'Close']))
index_pairs = list(set([(x[1], x[2]) for x in index.columns if x[0] == 'Close']))

cls_models = [RidgeClassifier]
reg_models = [SVR]
n_bins = 4

target_markets = ['BDT' ,'MNT', ('PKR', 'Karachi 100'), ('LKR', 'CSE All-Share')]
features = {"MNT": [None, "LKR", ("NZD", "NZX MidCap")],
            ('PKR', 'Karachi 100'): [None, "INR", ('JPY', 'NIkkei 225')],
            ('LKR', 'CSE All-Share'): [None, "IDR", ('MNT', 'MNE Top 20')],
            "BDT": [None, ("IDR", "IDX Composite"), "VND"]}

# forex_feature = "VND"
# index_feature = ("BDT", "DSE 30")

accuracy_lst = pd.DataFrame()
csv_name = ""
csv_dir = "./walk_forward_opt/"

metric = 'Close'
metrics = ['Open', 'Close', 'Low', 'High']
target = [metric + '_Ret']

forex_features = ['Intraday_HL', 'Intraday_OC', 'Prev_close_open'] + [y + x for x in
                                                                      ['_Ret', '_Ret_MA_3', '_Ret_MA_15', '_Ret_MA_45',
                                                                       '_MTD', '_YTD'] for y in metrics]
index_features = ['Intraday_HL', 'Intraday_OC', 'Prev_close_open'] + [y + x for x in
                                                                      ['_Ret', '_Ret_MA_3', '_Ret_MA_15', '_Ret_MA_45',
                                                                       '_MTD', '_YTD'] for y in (metrics)]

param_set = {"('PKR', 'Karachi 100')":{'alpha': 0.01, 'fit_intercept': True, 'normalize': False,
                                       'tol': 0.001, 'solver': 'sag', 'random_state': 1},
             "BDT": {'alpha': 10.0, 'fit_intercept': True, 'normalize': True, 'tol': 0.001,
                          'solver': 'sag', 'random_state': 4},
             "MNT": {'alpha': 10.0, 'fit_intercept': True, 'normalize': True, 'tol': 0.001,
                          'solver': 'sag', 'random_state': 4},
             "('LKR', 'CSE All-Share')":{'alpha': 0.001, 'fit_intercept': False, 'normalize': True,
                                         'tol': 0.0001, 'solver': 'auto', 'random_state': 1}}
# scalers = [None, MinMaxScaler, StandardScaler]
xstr = lambda s: 'None' if s is None else str(s)


def run_sklearn_model(model, train, test, feat, target, kwargs):
    X_train, y_train = train
    X_test, y_test = test
    X_train = pd.DataFrame(X_train)
    X_test = pd.DataFrame(X_test)
    period = len(X_train)
    y_train, y_test = y_train.astype(int), y_test.astype(int)
    prediction = []
    data = X_train.values
    data_y = y_train
    for i, t in enumerate(X_test.values):
        global accuracy_lst
        reg = model(**kwargs)
        reg.fit(data, data_y)
        if i == 0:
            y = reg.predict(X_train)
            for elem in y:
                prediction.append(elem)
        y = reg.predict([t])
        prediction.append(y[0])
        data = np.vstack((data, t))
        data = np.delete(data, 0, 0)
        data_y = np.append(data_y, y_test[i])
        data_y = np.delete(data_y, 0, 0)
    y_true = np.vstack((y_train, y_test))
    acc = accuracy_score(y_true, prediction)
    print(model.__name__ + " accurcy with " + xstr(feat) + " for window period " + str(period) + "=" + str(acc))
    return acc*100


def split_scale_kbins(X, y, scaler, train_index, test_index, shuffle=False, poly=False):
    X_train, X_test = X.iloc[:train_index], X.iloc[train_index:]
    y_train, y_test = y.iloc[:train_index], y.iloc[train_index:]
    if scaler is not None:
        scaler_X = scaler()
        if (scaler == FunctionTransformer):
            scaler_X = scaler(np.log1p)
        scaler_X = scaler_X.fit(X_train)
        X_train = scaler_X.transform(X_train)
        X_test = scaler_X.transform(X_test)

    scaler_y = KBinsDiscretizer(n_bins=n_bins, encode='ordinal', strategy='quantile')
    scaler_y.fit(y_test)
    y_train = scaler_y.transform(y_train)
    y_test = scaler_y.transform(y_test)
    y_test = np.array(y_test).reshape(-1, 1)
    y_train = np.array(y_train).reshape(-1, 1)
    return (X_train, X_test, y_train, y_test)


def get_sentiment(cur):
    sent_map = {'BDT': 'Dhaka',
                'MNT': 'UlaanBaatar',
                ('PKR', 'Karachi 100'): 'Karachi',
                ('LKR', 'CSE All-Share'): 'Colombo'}
    fill_vals = {'Avg': 0.5, 'Max': 0.5, 'Min': 0.5,
                 'Std': 0.0, 'Var': 0.0, 'Count': 0}
    f = lambda s: datetime.datetime.strptime(s, '%Y-%m-%d')
    ss = '#' + sent_map[cur] + '_aggregated_data.csv'
    sd = pd.read_csv(ss, header=0, parse_dates=[0], date_parser=f,
                     names=['Date', 'Avg', 'Max', 'Min', 'Std', 'Var', 'Count'])
    sd = sd.set_index(sd['Date']).drop(['Date'], axis=1)
    sd.index = sd.index.strftime("%Y-%m-%d")
    sd = sd.fillna(value=fill_vals)
    col_names = (pd.MultiIndex.from_tuples([('Sentiment', x) for x in sd.columns]))
    sd.columns = col_names
    return (sd)


def split_scale(X, y, scaler, train_index, test_index, shuffle=False, poly=False):
    X_train, X_test = X.iloc[:train_index], X.iloc[train_index:]
    y_train, y_test = y.iloc[:train_index], y.iloc[train_index:]
    if scaler is not None:
        scaler_X = scaler()
        if (scaler == FunctionTransformer):
            scaler_X = scaler(np.log1p)
        scaler_X = scaler_X.fit(X_train)
        X_train = scaler_X.transform(X_train)
        X_test = scaler_X.transform(X_test)

    scaler_y = Binarizer()
    y_train = np.array(y_train).reshape(1, -1)
    y_test = np.array(y_test).reshape(1, -1)
    y_train = scaler_y.transform(y_train)
    y_test = scaler_y.transform(y_test)
    y_train[y_train == 0] = -1
    y_test[y_test == 0] = -1
    y_test = np.array(y_test).reshape(-1, 1)
    y_train = np.array(y_train).reshape(-1, 1)
    return (X_train, X_test, y_train, y_test)


def add_cross_domain_features(feat):
    if (isinstance(feat, tuple)):
        index_cols = index_cols = [x for x in index.columns if x[1] == feat[0] and x[2] == feat[1]]
        X = index[[col for col in index_cols if col[0] in index_features + ['Time features']]][:-1]
    else:
        forex_cols = [x for x in forex.columns if x[1] == feat]
        X = forex[[col for col in forex_cols if col[0] in forex_features + ['Time features']]][:-1]
    return (X)


def do_forex(cur, model, train_index, test_index, feat, transf=None, shuffle=False, poly=False, kwargs={}):
    forex_cols = [x for x in forex.columns if x[1] == cur]
    X = forex[[col for col in forex_cols if col[0] in forex_features + ['Time features']]][:-1]
    if feat:
        X = X.join(add_cross_domain_features(feat))
    # X = X.join(get_sentiment(cur))
    y = forex[[col for col in forex_cols if col[0] in target]].shift(-1)[:-1]
    X = X.dropna(how='all', axis=1)
    X = X.dropna(how='any')
    y = y[y.index.isin(X.index)]
    X_train, X_test, y_train, y_test = split_scale(X, y, transf, train_index, test_index, shuffle, poly)
    # X_train, X_test = do_pca(X_train, X_test, X_train.columns)
    res = run_sklearn_model(model, (X_train, y_train), (X_test, y_test), feat, target, kwargs)
    return res


def do_index(cur, model, train_index, test_index, feat, transf=None, shuffle=False, poly=False, kwargs={}):
    index_cols = [x for x in index.columns if x[1] == cur[0] and x[2] == cur[1]]
    X = index[[col for col in index_cols if col[0] in index_features + ['Time features']]][:-1]
    if feat:
        X = X.join(add_cross_domain_features(feat))
    y = index[[col for col in index_cols if col[0] in target]].shift(-1)[:-1]
    X = X.dropna(how='all', axis=1)
    X = X.dropna(how='any')
    y = y[y.index.isin(X.index)]
    X_train, X_test, y_train, y_test = split_scale(X, y, transf, train_index, test_index, shuffle, poly)
    # X_train, X_test = do_pca(X_train,X_test,X_train.columns, int(X_train.shape[0]/5))
    res = run_sklearn_model(model, (X_train, y_train), (X_test, y_test), feat, target, kwargs)
    return (res)


def iterate_markets():
    shuffle = False
    poly = False
    global accuracy_lst
    reg_res = []
    for f_m in target_markets:
        print(f_m)
        for model in cls_models:
            print(model.__name__)
            if model == SVC:
                kwargs = {'kernel': 'rbf', 'C': 1.0, 'gamma': 'scale'}
                # feat = None
            elif model == RidgeClassifier:
                kwargs = param_set[f_m]
                # feat = 'VND'
            else:
                kwargs = {}
            for scaler in [None]:
                for feat in features[f_m]:
                    res = {}
                    rows = []
                    for i in range(30, 51, 5):
                        # try:
                        if f_m in forex_pairs:
                            train_index = i
                            test_index = 0
                            res[str(i)] = do_forex(f_m, model, train_index, test_index, feat, scaler, shuffle, poly,
                                                   kwargs)
                        else:
                            train_index = i
                            test_index = 0
                            res[str(i)] = do_index(f_m, model, train_index, test_index, feat, scaler, shuffle, poly,
                                                   kwargs)
                    reg_res = reg_res + [res]
    df = pd.DataFrame().append(reg_res)
    print(df)
    df.to_csv("%s_%s_window_accuracies.csv" % (csv_dir, cls_models[0].__name__))


iterate_markets()
