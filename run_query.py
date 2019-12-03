import re
import os
import sys
import shlex
import argparse
import subprocess
from datetime import datetime, timedelta

parser = argparse.ArgumentParser(description='Script to update twitter queries')
parser.add_argument('--queries', action="store", default="twitter_query.txt", dest="filename", help="Enter file name")
file_name = parser.parse_args().filename


with open(file_name, 'r') as f:
    queries = f.readlines()

while(True):
    q = queries[0].strip('\n')
    queries = queries[1:]
    print("Checking query "+q)
    if(q[0] in ['*', '&']):
        queries.append(q+'\n')
        continue
    else:
        print("The query to be run is " + q[2:] + "...")
        query = q[3:].split(' ')[0]
        os.system('pwd')
        os.system('cd TweetScraper-master \n nohup scrapy crawl TweetScraper -a query=\"'+q[2:]+'\" > Data/log/nohup'+query+'.out &')
        print("Running query...")
        q = '*' + q[1:]
        queries.append(q+'\n')

        with open(file_name, 'w') as f:
            for q_1 in queries:
                f.write(q_1)
        break
