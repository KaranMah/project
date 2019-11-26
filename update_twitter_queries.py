import re
import os
import sys
import subprocess
from datetime import datetime, timedelta

from_date = "2010-01-01"
today = datetime.today().strftime("%Y-%m-%d")
to_date = "2019-12-31"

if(datetime.strptime(today, "%Y-%m-%d") < datetime.strptime(to_date, "%Y-%m-%d")):
    to_date = today

with open('twitter_query.txt', 'r') as f:
    queries = f.readlines()

queries = [x.strip() for x in queries]
new_queries = []
for query in queries:
    query_file = '_'.join(query.replace(':', ' ').split(' '))
    if(os.path.exists("TweetScraper-master/Data/tweet/"+query_file)):
        out = os.popen('tail '+"TweetScraper-master/Data/tweet/"+query_file+' -c 200')
        oldest = re.search(r"[0-9]{4}-[0-9]{2}-[0-9]{2}", out.read()).group()
        if(datetime.strptime(oldest, "%Y-%m-%d") > datetime.strptime(from_date, "%Y-%m-%d")):
            new_q = ':'.join(query.split(':')[:-1]) + ":" + datetime.strftime(datetime.strptime(oldest, "%Y-%m-%d")-timedelta(days=1), "%Y-%m-%d")
            new_queries.append(new_q)
        if(datetime.strptime(query.split(':')[-1], "%Y-%m-%d") < (datetime.strptime(to_date, "%Y-%m-%d")-timedelta(days=1))):
            new_q = query.split(':')[0] + ":" + datetime.strftime(datetime.strptime(query.split(':')[-1], "%Y-%m-%d")+timedelta(days=1), "%Y-%m-%d") + " until:" +  datetime.strftime(datetime.strptime(today, "%Y-%m-%d")-timedelta(days=1), "%Y-%m-%d")
            new_queries.append(new_q)
    elif(os.path.exists("/data/"+query_file)):
        out = os.popen('tail '+"/data/"+query_file+' -c 200')
        oldest = re.search(r"[0-9]{4}-[0-9]{2}-[0-9]{2}", out.read()).group()
        if(datetime.strptime(oldest, "%Y-%m-%d") > datetime.strptime(from_date, "%Y-%m-%d")):
            new_q = ':'.join(query.split(':')[:-1]) + ":" + datetime.strftime(datetime.strptime(oldest, "%Y-%m-%d")-timedelta(days=1), "%Y-%m-%d")
            new_queries.append(new_q)
        if(datetime.strptime(query.split(':')[-1], "%Y-%m-%d") < (datetime.strptime(to_date, "%Y-%m-%d")-timedelta(days=1))):
            new_q = query.split(':')[0] + ":" + datetime.strftime(datetime.strptime(query.split(':')[-1], "%Y-%m-%d")+timedelta(days=1), "%Y-%m-%d") + " until:" +  datetime.strftime(datetime.strptime(today, "%Y-%m-%d")-timedelta(days=1), "%Y-%m-%d")
            new_queries.append(new_q)
    else:
        if(datetime.strptime(query.split(':')[-1], "%Y-%m-%d") < (datetime.strptime(to_date, "%Y-%m-%d")-timedelta(days=1))):
            new_q = query.split(':')[0] + ":" + datetime.strftime(datetime.strptime(query.split(':')[-1], "%Y-%m-%d")+timedelta(days=1), "%Y-%m-%d") + " until:" +  datetime.strftime(datetime.strptime(today, "%Y-%m-%d")-timedelta(days=1), "%Y-%m-%d")
            new_queries.append(new_q)
        else:
            new_queries.append(query)

with open('twitter_query_v2.txt', 'w') as f:
    for query in new_queries:
        f.write(query)
        f.write('\n')
f.close()
