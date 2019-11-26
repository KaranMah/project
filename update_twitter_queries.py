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
    print(query)
    print(query_file)
    print()
    if(os.path.exists("TweetScraper-master/Data/tweet/"+query_file)):
        out = subprocess.Popen(['tail '+"TweetScraper-master/Data/tweet/"+query_file+' -c 200'],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT)
        stdout, stderr = out.communicate()
        oldest = re.search(r"[0-9]{4}-[0-9]{2}-[0-9]{2}", stdout).group()
        if(datetime.strptime(oldest, "%Y-%m-%d") > datetime.strptime(from_date, "%Y-%m-%d")):
            new_q = query.split(':')[:-1] + ":" + datetime.strftime(datetime.strptime(oldest, "%Y-%m-%d")-timedelta(days=1), "%Y-%m-%d")
            new_queries.append(new_q)
        if(datetime.strptime(query.split(':')[-1], "%Y-%m-%d") < datetime.strptime(to_date, "%Y-%m-%d")):
            new_q = query.split(':')[0] + ":" + datetime.strftime(datetime.strptime(query.split(':')[-1], "%Y-%m-%d")+timedelta(days=1), "%Y-%m-%d") + " until:" +  datetime.strftime(datetime.strptime(today, "%Y-%m-%d")-timedelta(days=1), "%Y-%m-%d")
    else:
        pass
        new_queries.append(query)
print(new_queries)
