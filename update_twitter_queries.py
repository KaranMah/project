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

yesterday = datetime.today()-timedelta(days=1)

queries = [x.strip() for x in queries]
new_queries = queries
for query in queries:
    temp_query = query.strip("?!*&$")
    filename = '_'.join(temp_query.replace(':', ' ').split(' '))
    if(os.path.exists("TweetScraper/Data/tweet/"+temp_query)):
        query = '*' + query[1:]
    elif(os.path.exists("/data/"+temp_query)):
        query = '&' + query[1:]
    last_date = datetime.strptime(query.split(' ')[2].split(':')[1], "%Y-%m-%d")+timedelta(days=1)
    if((yesterday-last_date).days > 0):
        new_q = "?!" + query[2:].split(' ')[0] + " since:"+datetime.strptime(last_date, "%Y-%m-%d") + " until:" + datetime.strptime(yesterday, "%Y-%m-%d")
    if(query[1] == '$'):
        pass
    elif(query[0] != '?'):
        new_queries.append(new_q)
    else:
        new_q = query.split(' ')[0] + query.split(' ')[1] + " until:" + datetime.strptime(yesterday, "%Y-%m-%d")
        new_queries.remove(query)
        new_queries.append(new_q)


with open(file[:-4]+'_v2.txt', 'w') as f:
    for query in new_queries:
        f.write(query)
        f.write('\n')
f.close()

print("Queries have been updated!")
print("Key: ")
print("! Incumbent query")
print("$ Fixed date query")
print("* Running query")
print("& Done query")
print("? Yet to run query")
