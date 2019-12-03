import re
import os
import sys
import argparse
import subprocess
from datetime import datetime, timedelta

parser = argparse.ArgumentParser(description='Script to update twitter queries')
parser.add_argument('--queries', action="store", default="twitter_query.txt", dest="filename", help="Enter file name")
file_name = parser.parse_args().filename

with open(file_name, 'r') as f:
    queries = f.readlines()

yesterday = datetime.today()-timedelta(days=1)

queries = [x.strip('\n') for x in queries]
new_queries = []
for query in queries:
    temp_query = query.strip("?!*&$")
    filename = '_'.join(temp_query.replace(':', ' ').split(' '))
    if(os.path.exists("TweetScraper/Data/tweet/"+filename)):
        query = '*' + query[1:]
    elif(os.path.exists("/data/"+filename)):
        query = '&' + query[1:]
    last_date = datetime.strptime(query.split(' ')[2].split(':')[1], "%Y-%m-%d")+timedelta(days=1)
    if(query[1] == '!' and (yesterday-last_date).days > 0):
        new_q = "?!" + query[2:].split(' ')[0] + " since:"+datetime.strftime(last_date, "%Y-%m-%d") + " until:" + datetime.strftime(yesterday, "%Y-%m-%d")
    if(query[1] == '$'):
        new_queries.append(query)
    elif(query[0] != '?'):
        new_queries.append(query)
        new_queries.append(new_q)
    else:
        new_q = query.split(' ')[0] +" " + query.split(' ')[1] + " until:" + datetime.strftime(yesterday, "%Y-%m-%d")
        new_queries.append(new_q)


with open(file_name[:-4]+'_v3.txt', 'w') as f:
    for query in new_queries:
        f.write(query)
        f.write('\n')
f.close()

print()
print("**************************")
print("Queries have been updated!")
print("Key: ")
print("! Incumbent query")
print("$ Fixed date query")
print("* Running query")
print("& Done query")
print("? Yet to run query")
print("******************")
