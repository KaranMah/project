import ast
import json
import math
import pprint
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D

%matplotlib inline

files = ['optimization_results/sec_opt_RidgeClassifier_BDT.csv',
         'optimization_results/sec_opt_SVC_BDT.csv']

cols = ['accuracy', 'args', 'target_market', 'feature_used']

def Ridge(file):
    data = pd.read_csv(file, names = cols, header=0,
                       converters = {'args': ast.literal_eval}).reset_index(drop=True)

    data = pd.concat([data, pd.json_normalize(data['args'])], axis=1)
    data = data.drop(['args', 'target_market', 'feature_used'], axis=1)
    vars = ['alpha', 'fit_intercept', 'normalize', 'tol', 'solver']
    nd = data.groupby(vars, as_index=False)['accuracy'].mean()
    nd = nd[nd['fit_intercept'] == True]
    nd = nd[nd['normalize'] == False]
    nd = nd.drop(['fit_intercept', 'normalize'], axis=1)
    print(nd)
    fig = plt.figure()
    ax = Axes3D(fig)
    surf = ax.plot_trisurf(nd['alpha'], nd['tol'], nd['accuracy'], cmap = cm.coolwarm,
                       linewidth=0, antialiased=False)
    plt.title("RidgeClassifier BDT Optimization")
    ax.set_xlabel('Alpha',rotation=150)
    ax.set_ylabel('Tolerance')
    ax.set_zlabel('Accuracy')
    plt.show()


def SVC(file):
    data = pd.read_csv(file, names = cols, header=0,
                       converters = {'args': ast.literal_eval}).reset_index(drop=True)

    data = pd.concat([data, pd.json_normalize(data['args'])], axis=1)
    data = data.drop(['args', 'target_market', 'feature_used'], axis=1)
    vars = ['C', 'gamma', 'kernel', 'tol']
    nd = data[data['gamma'] == 'auto']
    print(nd)
    fig = plt.figure()
    ax = Axes3D(fig)
    surf = ax.plot_trisurf(nd['C'], nd['tol'], nd['accuracy'], cmap = cm.coolwarm,
                       linewidth=0, antialiased=False)


    plt.show()

Ridge(files[0])
# SVC(files[1])
