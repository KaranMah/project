import json
import numpy as np
import pandas as pd
import datetime as datetime

config = {
    "leaders_data": "./social_media.txt",
    "earliest_date" : "01/01/2010",
    "latest_date" : '31/10/2019'	#datetime.datetime.now().strftime("%d/%m/%Y")
}

def read_file(filename):
    dataset = pd.read_csv(filename, sep=';')
    dataset['HOSS'] = dataset['HOSS'].apply(lambda x: x.split(','))
    return dataset

def parsetime(time):
    temp_time = time.split('/')
    return temp_time[2] + '-' + temp_time[1] + '-' + temp_time[0]

def twitter_query(hashtag, from_date, to_date):
    if(datetime.datetime.strptime(from_date, "%d/%m/%Y") < datetime.datetime.strptime(config['earliest_date'], "%d/%m/%Y")):
        from_date = config['earliest_date']
    if(to_date == 'Incumbent'):
        to_date = config['latest_date']
    return "#" +  "".join(hashtag) + " since:" + parsetime(from_date) + " until:" + parsetime(to_date)

def gen_twitter_queries(data):
    queries = []
    for index, row in data.iterrows():
        queries.append(twitter_query(row["Country"], config['earliest_date'], config['latest_date']))
        queries.append(twitter_query(row["City"], config['earliest_date'], config['latest_date']))
        tp = row['HOSS']
        queries += [ twitter_query(x.split('(')[0], x.split('(')[1].split('-')[0], x.split('(')[1].split('-')[1][:-1]) for x in tp ]
    with open('twitter_query.txt', 'w') as f:
        for item in queries:
            f.write("%s\n" % item)

def gen_reddit_queries(data):
    queries = []
    for index, row in data.iterrows():
        queries.append("".join(row['Country'].split(' ')).lower())
        queries.append("".join(row['City'].split(' ')).lower())
    with open('reddit_query.txt', 'w') as f:
        for item in queries:
            f.write("%s\n" % item)

data = read_file(config['leaders_data'])
gen_reddit_queries(data)
