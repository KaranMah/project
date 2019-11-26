import re
import os
import sys
import argparse
import subprocess
from datetime import datetime, timedelta

parser = argparse.ArgumentParser(description='Script to update twitter queries')
parser.add_argument('--queries', action="store", default="twitter_query.txt", dest="filename", help="Enter file name")
file = parser.parse_args().filename

with open(file, 'r') as f:
    queries = f.readlines()

queries = [x.strip() for x in queries]
new_queries = []
for query in queries:
    done = False
    from_date = "2010-01-01"
    today = datetime.today().strftime("%Y-%m-%d")
    to_date = "2020-01-01"

    if(datetime.strptime(today, "%Y-%m-%d") < datetime.strptime(to_date, "%Y-%m-%d")):
        to_date = today

    if(query[0] == '!'):
        from_date = (query.split(' ')[1]).split(':')[1]

    if(query[0] =='$'):
        from_date = query.split(' ')[1].split(':')[1]
        to_date = datetime.strftime(datetime.strptime(query.split(' ')[1].split(':')[1], "%Y-%m-%d")+timedelta(days=1), "%Y-%m-%d")

    query_file = '_'.join(query.strip("*!$").replace(':', ' ').split(' '))
    if(os.path.exists("/data/"+query_file)):
        out = os.popen('tail '+"/data/"+query_file+' -c 200')
        oldest = re.search(r"[0-9]{4}-[0-9]{2}-[0-9]{2}", out.read()).group()
        if(datetime.strptime(oldest, "%Y-%m-%d") > datetime.strptime(from_date, "%Y-%m-%d")):
            new_q = ':'.join(query.split(':')[:-1]) + ":" + datetime.strftime(datetime.strptime(oldest, "%Y-%m-%d")-timedelta(days=1), "%Y-%m-%d")
            new_queries.append('$' + new_q)
            done = True
        if(datetime.strptime(query.split(':')[-1], "%Y-%m-%d") < (datetime.strptime(to_date, "%Y-%m-%d")-timedelta(days=1))):
            new_q = query.split(':')[0] + ":" + datetime.strftime(datetime.strptime(query.split(':')[-1], "%Y-%m-%d")+timedelta(days=1), "%Y-%m-%d") + " until:" +  datetime.strftime(datetime.strptime(today, "%Y-%m-%d")-timedelta(days=1), "%Y-%m-%d")
            new_queries.append(new_q)
            done = True
    else:
        if(datetime.strptime(query.split(':')[-1], "%Y-%m-%d") < (datetime.strptime(to_date, "%Y-%m-%d")-timedelta(days=1))):
            new_q = query.split(':')[0] + ":" + datetime.strftime(datetime.strptime(query.split(':')[1].split(' ')[0], "%Y-%m-%d"), "%Y-%m-%d") + " until:"
            if(query[0] == '$'):
                new_q += query.split(':')[-1]
            else:
                new_q += datetime.strftime(datetime.strptime(today, "%Y-%m-%d")-timedelta(days=1), "%Y-%m-%d")
            new_queries.append(new_q)
            done = True
        else:
            new_queries.append(query)
    if(done==False):
        new_queries.append('*' + query)

with open(file[:-4]+'_v2.txt', 'w') as f:
    for query in new_queries:
        f.write(query)
        f.write('\n')
f.close()

print("Queries have been updated!")
print("Key: ")
print("! Incumbent Head of state")
print("$ Previous Head of state")
print("* Completed query")
