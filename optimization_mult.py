import pandas as pd
import numpy as np
from sklearn.linear_model import *
from sklearn.metrics import *
from sklearn.preprocessing import *
from sklearn.svm import *
import warnings
from threading import Thread, Lock
from sklearn.exceptions import DataConversionWarning


warnings.filterwarnings(action='ignore', category=DataConversionWarning)
warnings.filterwarnings(action='ignore', category=UserWarning)

forex = pd.read_csv('prep_forex.csv', header=[0, 1], index_col=0)
index = pd.read_csv('prep_index.csv', header=[0, 1, 2], index_col=0)

forex_pairs = list(set([x[1] for x in forex.columns if x[0] == 'Close']))
index_pairs = list(set([(x[1], x[2]) for x in index.columns if x[0] == 'Close']))

cls_models = [SVC]

target_markets = ['MNT', 'BDT', ('LKR', 'CSE All-Share')]
features = {"MNT": [None, "LKR", ("NZD", "NZX MidCap")],
            ('PKR', 'Karachi 100'): [None, "INR", ('JPY', 'NIkkei 225')],
            ('LKR', 'CSE All-Share'): [None, "IDR", ('MNT', 'MNE Top 20')],
            "BDT": [("IDR", "IDX Composite"), None, "VND"]}

kernel = ['linear', 'rbf']
C = [.1,.2,.3,.4,.5,.6,.7,.8,.9,1.0]
gamma = ['auto', 'scale']

result = (0, None, None, None)
result_lock = Lock()
columns=['accuracy', 'args', 'target_market', 'feature_used']
result_df = pd.DataFrame(columns=columns)

metric = 'Close'
metrics = ['Open', 'Close', 'Low', 'High']
target = [metric + '_Ret']

forex_features = ['Intraday_HL', 'Intraday_OC', 'Prev_close_open'] + [y + x for x in
                                                                      ['_Ret', '_Ret_MA_3', '_Ret_MA_15', '_Ret_MA_45',
                                                                       '_MTD', '_YTD'] for y in metrics]
index_features = ['Intraday_HL', 'Intraday_OC', 'Prev_close_open'] + [y + x for x in
                                                                      ['_Ret', '_Ret_MA_3', '_Ret_MA_15', '_Ret_MA_45',
                                                                       '_MTD', '_YTD'] for y in (metrics + ['Volume'])]

xstr = lambda s: '' if s is None else str(s)


def run_sklearn_model(model, train, test, feat, kwargs):
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
    prediction = pd.DataFrame(prediction)

    acc = accuracy_score(y_true, prediction)
    #print("accurcy for " + xstr(feat) + " with period " + str(period)+ " and params " + kwargs +"="+str(acc))
    return acc


def split_scale(X, y, scaler, train_index, test_index):
    X_train, X_test = X.iloc[:train_index], X.iloc[train_index:]
    y_train, y_test = y.iloc[:train_index], y.iloc[train_index:]
    if scaler is not None:
        scaler_X = scaler()
        if(scaler == FunctionTransformer):
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
    if(isinstance(feat, tuple)):
        index_cols = [x for x in index.columns if x[1] == feat[0] and x[2] == feat[1]]
        X = index[[col for col in index_cols if col[0] in index_features + ['Time features']]][:-1]
    else:
        forex_cols = [x for x in forex.columns if x[1] == feat]
        X = forex[[col for col in forex_cols if col[0] in forex_features + ['Time features']]][:-1]
    return(X)


def do_forex(cur, model, train_index, test_index, feat, transf, kwargs):
    forex_cols = [x for x in forex.columns if x[1] == cur]
    X = forex[[col for col in forex_cols if col[0] in forex_features + ['Time features']]][:-1]
    if feat:
        X = X.join(add_cross_domain_features(feat))

    y = forex[[col for col in forex_cols if col[0] in target]].shift(-1)[:-1]
    X = X.dropna(how='all', axis=1)
    X = X.dropna(how='any')
    y = y[y.index.isin(X.index)]

    X_train, X_test, y_train, y_test = split_scale(X, y, transf, train_index, test_index)
    res = run_sklearn_model(model, (X_train, y_train), (X_test, y_test), feat, kwargs)
    return res


def do_index(cur, model, train_index, test_index, feat, transf, kwargs):
    index_cols = [x for x in index.columns if x[1] == cur[0] and x[2] == cur[1]]
    X = index[[col for col in index_cols if col[0] in index_features + ['Time features']]][:-1]
    if feat:
        X = X.join(add_cross_domain_features(feat))
    y = index[[col for col in index_cols if col[0] in target]].shift(-1)[:-1]
    X = X.dropna(how='all', axis=1)
    X = X.dropna(how='any')
    y = y[y.index.isin(X.index)]
    X_train, X_test, y_train, y_test = split_scale(X, y, transf, train_index, test_index)
    res = run_sklearn_model(model, (X_train, y_train), (X_test, y_test), feat, kwargs)
    return res


def iterate_markets(model, f_m, feat, kwargs):
    global result
    global result_df
    reg_res = (0, None, None, None)
    for i in range(30, 55, 5):
        try:
            if f_m in forex_pairs:
                train_index = i
                test_index = 0
                res = do_forex(f_m, model, train_index, test_index, feat, None, kwargs)
            else:
                train_index = i
                test_index = 0
                res = do_index(f_m, model, train_index, test_index, feat, None, kwargs)

            if reg_res[0] < res:
                reg_res = (res, kwargs, f_m, feat)

        except Exception as e:
            pass
    with result_lock:
        if result[0] < reg_res[0]:
            result = reg_res
        result_df = pd.concat([result_df, pd.DataFrame([reg_res], columns=columns)])


def main():
    numTot = len(cls_models) * len(target_markets) * len(features[target_markets[0]])
    global result_df
    params = []
    result = (0,None, None, None)
    for k in kernel:
        for c in C:
            for g in gamma:
                params.append({'kernel': k, 'C': c, 'gamma': g})

    print(len(params))
    threads = []
    for f in target_markets:
        for model_name in cls_models:
            for feature in features[f]:
                for ind, p in enumerate(params):
                    try:
                        threads.append(Thread(target=iterate_markets, args=(model_name, f, feature, p)))
                    except Exception as e:
                        print("main, load ", e)
        print(len(threads))

        for thread in threads:
            try:
                thread.start()
            except Exception as e:
                print("main, start ", e)

        for thread in threads:
            thread.join()

        threads = []

        print("best score "+f+" =", result)
        result_df.to_csv("./optimization_results/optimization_mult_"+f+".csv")
        result_df = pd.DataFrame(columns=columns)
    
main()







