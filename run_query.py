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

while(True):
    q = queries.pop()
    if(q[0] in "*&"):
        queries.append(q)
        continue
    else:
        print("The query to be run is " + q + "...")
        query = q[2:].split(' ')[0]
        os.popen('nohup scrapy crawl TweetScraper -a query="'+q[2:]+'" & >  /Data/log/nohup'+query+'.out &')
        print("Running query...")
        q = '*' + q[1:]
        queries.append(q)
        break
