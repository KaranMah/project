import json
import investpy
import numpy as np
import pandas as pd


config = {
    "filepath": "./currencies.txt"
}

def read_file(filename):
    dataset = pd.read_csv(filename, sep=';')
    dataset["Index"] = dataset["Index"].apply(lambda x: x.split(','))
    return dataset


def get currency_pairs(currencies):
    ## TODO:
    data = investpy.get_currency_cross_recent_data(currency_cross='EUR/USD')

# if __name__ == '__main__':
data = read_file(config["filepath"])
cur_list = list(data["Currency"])
