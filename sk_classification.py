import json
import math
import pprint
import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

%matplotlib inline

from sklearn.decomposition import PCA
from sklearn.linear_model import *
from sklearn.preprocessing import *
from sklearn.experimental import enable_hist_gradient_boosting
from sklearn.model_selection import *
from sklearn.kernel_ridge import *
from sklearn.svm import *
from sklearn.neighbors import *
from sklearn.gaussian_process import *
from sklearn.cross_decomposition import *
from sklearn.naive_bayes import *
from sklearn.tree import *
from sklearn.experimental import *
from sklearn.ensemble import *
from sklearn.metrics import *

forex = pd.read_csv('prep_forex.csv', header=[0,1], index_col=0)
index = pd.read_csv('prep_index.csv', header=[0,1,2], index_col=0)

forex_pairs = list(set([x[1] for x in forex.columns if x[0] == 'Close']))
index_pairs = list(set([(x[1], x[2]) for x in index.columns if x[0] == 'Close']))

# reg_models = [LinearRegression, Ridge, RidgeCV, Lasso, LassoCV, LassoLarsCV, LassoLarsIC, MultiTaskLasso,
#               ElasticNet, ElasticNetCV, MultiTaskElasticNet, MultiTaskElasticNetCV, LassoLars,
#               OrthogonalMatchingPursuit, BayesianRidge, ARDRegression,
#               SGDRegressor, PassiveAggressiveRegressor, HuberRegressor, RANSACRegressor, TheilSenRegressor,
#               KernelRidge, SVR, NuSVR, KNeighborsRegressor, RadiusNeighborsRegressor,
#               GaussianProcessRegressor, PLSRegression, DecisionTreeRegressor,
#               BaggingRegressor, AdaBoostRegressor, ExtraTreesRegressor, GradientBoostingRegressor,
#               RandomForestRegressor, HistGradientBoostingRegressor]

cls_models = [RidgeClassifier, LogisticRegression,  LogisticRegressionCV,
              SGDClassifier, Perceptron, PassiveAggressiveClassifier, SVC, NuSVC, LinearSVC,
              KNeighborsClassifier, NearestCentroid, GaussianProcessClassifier,
              GaussianNB, MultinomialNB, ComplementNB, BernoulliNB, CategoricalNB,
              DecisionTreeClassifier, BaggingClassifier, AdaBoostClassifier, ExtraTreesClassifier,
              GradientBoostingClassifier, RandomForestClassifier, HistGradientBoostingClassifier]



metric = 'Close'
metrics = ['Open', 'Close', 'Low', 'High']
target = [metric+'_Ret']
forex_features = ['Intraday_HL', 'Intraday_OC', 'Prev_close_open'] + [y+x for x in ['_Ret', '_Ret_MA_3', '_Ret_MA_15', '_Ret_MA_45', '_MTD', '_YTD'] for y in metrics]
index_features = ['Intraday_HL', 'Intraday_OC', 'Prev_close_open'] + [y+x for x in ['_Ret', '_Ret_MA_3', '_Ret_MA_15', '_Ret_MA_45', '_MTD', '_YTD'] for y in (metrics + ['Volume'])]
n_bins = 5
scalers = [Binarizer, KBinsDiscretizer]

def plot_results(y_true, y_pred, model):
    plot_df = pd.concat([y_true.reset_index(drop=True), pd.DataFrame(y_pred)], axis=1, ignore_index=True)
    plt.figure()
    plt.plot(plot_df)
    plt.title(model().__class__.__name__)


def save_res_to_csv(data):
    data.to_csv(cur+"_"+mod+".csv")

def add_cross_domain_features(feat):
    if(isinstance(feat, tuple)):
        index_cols = index_cols = [x for x in index.columns if x[1] == feat[0] and x[2] == feat[1]]
        X = index[[col for col in index_cols if col[0] in index_features + ['Time features']]][:-1]
    else:
        forex_cols = [x for x in forex.columns if x[1] == feat]
        X = forex[[col for col in forex_cols if col[0] in forex_features + ['Time features']]][:-1]
    return(X)

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

def run_sklearn_model(model, train, test, features, target):
    try:
        X_train, y_train = train
        X_test, y_test = test
    except:
        pass
    reg = model()#kernel='poly', degree=5)#penalty='elasticnet')#solver='liblinear', max_iter=1000)#, tol=1e-3)
    # print(X_train)
    # print(y_train)
    reg.fit(X_train, y_train)
    # print(reg.coef_)
    try:
        pprint.pprint(dict(zip(X_train.columns.values,reg.feature_importances_)))
    except:
        pass
    y_pred = reg.predict(X_test)
    save_data = pd.concat([pd.DataFrame(y_test), pd.DataFrame(y_pred)], axis=1,
                          ignore_index=True)
    # save_data.index = X_test.index
    save_data.columns = ['Real', 'Pred']
    print(save_data)
    # save_res_to_csv(save_data)
    print(classification_report(y_test, y_pred))
    plot_confusion_matrix(reg, X = X_test, y_true = y_test,
                          display_labels=np.unique(y_test), values_format = '.5g')
    try:
        return({"F1" :f1_score(y_test, y_pred, average='weighted'),
            "Precision": precision_score(y_test, y_pred, average='weighted'),
            "Recall": recall_score(y_test, y_pred, average='weighted'),
            "AUC": roc_auc_score(y_test, reg.fit(X_train, y_train).decision_function(X_test), multi_class='ovr')})
    except:
        return({"F1" :f1_score(y_test, y_pred, average='weighted'),
            "Precision": precision_score(y_test, y_pred, average='weighted'),
            "Recall": recall_score(y_test, y_pred, average='weighted'),
            "Accuracy": accuracy_score(y_test, y_pred)})
    #     y_pred = ([x[0] for x in y_pred])
    #     for i in range(len(y_pred)):
    #         if(np.isnan(y_pred[i])):
    #             y_pred[i] = 0
    #     return({"F1" :f1_score(y_test, y_pred, average='weighted'),
    #         "Precision": precision_score(y_test, y_pred),
    #         "Recall": recall_score(y_test, y_pred),
    #         "AUC": roc_auc_score(y_test, reg.fit(X_train, y_train).decision_function(X_test))})
    # print("MSE: ", mean_squared_error(y_test, y_pred))
    # print("R2: ", r2_score(y_test, y_pred))
    # plot_results(y_test, y_pred, model)
def do_pca(train, test, names, n=None):
    pca = PCA(n).fit(train)
    plt.plot(np.cumsum(pca.explained_variance_ratio_))
    plt.xlabel('Number of components')
    plt.ylabel('Cumulative explained variance')
    plt.show()
    res = (dict(zip(names, pca.explained_variance_ratio_)))
    sorted_res = (sorted(res.items(), key=lambda x: x[1], reverse=True))
    print(sorted_res)
    X_train = (pca.transform(train))
    X_test = (pca.transform(test))
    return(X_train, X_test)


def split_scale(X, y, scaler, shuffle=False, poly=False, transf_features_also=False):
    X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=shuffle, test_size=0.2)
    if(poly):
        X_train = PolynomialFeatures(2).fit_transform(X_train)
        X_test = PolynomialFeatures(2).fit_transform(X_test)
    if (scaler == scalers[-1]):
        scaler_y = scaler(n_bins=n_bins, encode='ordinal', strategy='quantile')
    else:
        scaler_y = scaler()
    scaler_y = scaler_y.fit(y_train)
    # print("~~~~~~")
    # print(scaler_y.bin_edges_)
    y_train = scaler_y.transform(y_train)
    y_test = scaler_y.transform(y_test)
    y_train[y_train == 0] = -1
    y_test[y_test == 0] = -1
    return(X_train, X_test, y_train.astype(int), y_test.astype(int))


def do_forex(cur, model, feat=None, transf = None, shuffle=False, poly=False, transf_features_also=False):
    forex_cols = [x for x in forex.columns if x[1] == cur]
    X = forex[[col for col in forex_cols if col[0] in forex_features + ['Time features']]][:-1]
    y = forex[[col for col in forex_cols if col[0] in target]].shift(-1)[:-1]
    if feat:
        X = X.join(add_cross_domain_features(feat))
    sent = get_sentiment(cur)
    X = X.join(sent, how='left')
    X = X.dropna(how='any')
    y = y[y.index.isin(X.index)]
    X_train, X_test, y_train, y_test = split_scale(X, y, transf, shuffle, poly, transf_features_also)
    X_train, X_test = do_pca(X_train, X_test, X.columns, 20)
    res = run_sklearn_model(model, (X_train, y_train), (X_test, y_test), forex_features, target)
    return(res)

def do_index(cur, model, transf = None, shuffle=False, poly=False, transf_features_also=False):
    index_cols = [x for x in index.columns if x[1] == cur[0] and x[2] == cur[1]]
    X = index[[col for col in index_cols if col[0] in index_features + ['Time features']]][:-1]
    y = index[[col for col in index_cols if col[0] in target]].shift(-1)[:-1]
    X = X.dropna(how='any')
    y = y[y.index.isin(X.index)]
    X_train, X_test, y_train, y_test = split_scale(X, y, transf, shuffle, poly, transf_features_also)
    res = run_sklearn_model(model, (X_train, y_train), (X_test, y_test), index_features, target)
    return(res)

def iterate_markets():
    reg_res = []
    for f_m in ['MNT', 'BDT', ('PKR', 'Karachi 100'), ('LKR', 'CSE All-Share')]:#(forex_pairs+index_pairs):
        print(f_m)
        for model in cls_models:
            for scaler in scalers:
                for shuffle in [True, False]:
                    for poly in [True, False]:
                        for transf_features_also in [True, False]:
                            try:
                                if(f_m in forex_pairs):
                                    res = do_forex(f_m, model, scaler, shuffle, poly, transf_features_also)
                                else:
                                    res = do_index(f_m, model, scaler, shuffle, poly, transf_features_also)
                            except:
                                #res = do_forex(f_m, model, scaler, shuffle, poly, transf_features_also)
                                continue
                            res['Pair'] = f_m
                            res['Transformation'] = scaler().__class__.__name__ if scaler is not None else None
                            res['Shuffle'] = shuffle
                            res['Model'] = model().__class__.__name__
                            res['Poly'] = poly
                            res['Features transformed'] = transf_features_also
                            print(res)
                            reg_res.append(res)
    return(reg_res)

# res = iterate_markets()
# res_df = pd.DataFrame(res, columns= ['Pair', 'Model', 'Transformation', 'Shuffle', 'Poly', 'Features transformed', 'F1', 'Precision', 'Recall', 'AUC'])
# print(res_df)
# res_df.to_csv("sk_classification.csv")
# do_stuff(["HKD", "Hang Seng"], LinearRegression)
cur = 'BDT'
mod = RandomForestClassifier
do_forex(cur, mod, None, Binarizer, False, False, False)
# do_index(('PKR', 'Karachi 100'), RandomForestClassifier, Binarizer, False, False, False)
