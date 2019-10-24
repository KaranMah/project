import json
import investpy
import numpy as np
import pandas as pd
import datetime as datetime

config = {
    "raw_data": "./currencies.txt",
    "cur_pairs": "./cur_pairs.txt"
}

def read_file(filename):
    dataset = pd.read_csv(filename, sep=';')
    dataset["Index"] = dataset["Index"].apply(lambda x: x.split(','))
    return dataset


def get_currency_pairs(currencies):
    lol = [(x + "/" + currencies[0]) for x in currencies[1:]]
    with open('cur_pairs.txt', 'w') as f:
        for item in lol:
            f.write("%s\n" % item)


def get_forex_rates(filename):
    earliest_date = '01/01/2010'
    latest_date = datetime.datetime.now().strftime("%d/%m/%Y")
    pairs = list(pd.read_csv(filename, header=None)[0])
    for pair in pairs:
        df = investpy.get_currency_cross_historical_data(currency_cross='EUR/USD',
                                                 from_date=earliest_date,
                                                 to_date=latest_date)
        print(df.shape)

# if __name__ == '__main__':
data = read_file(config["raw_data"])
get_currency_pairs(list(data["Currency"]))
get_forex_rates(config["cur_pairs"])
