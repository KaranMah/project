import json
import investpy
import numpy as np
import pandas as pd


config = {
    "filepath": "./currencies.txt"
}

def read_file(filename):
    dataset = pd.read_csv(filename, sep=';', delimiter='!', header=None)
    return dataset


if __name__ == '__main__':
    data = read_file(config["filepath"])
    data
