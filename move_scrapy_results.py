import re
import os
import sys
import argparse
import subprocess
from datetime import datetime, timedelta

parser = argparse.ArgumentParser(description='Script to move Scrapy results')
parser.add_argument('--source', action="store", default="TweetScraper-master/Data/tweet", dest="source", help="Enter directory of files")
parser.add_argument('--log', action="store", default="TweetScraper-master/Data/log", dest="log", help="Enter directory containing nohup files")
source = parser.parse_args().source
log = parser.parse_args().log

with open
('twitter_query.txt', 'r') as f:
    queries = f.readlines()

print(dest)
fileobj = os.popen('ls '+log+' | grep -E "nohup"')
files = fileobj.read()
files_array = files.strip('\n').split('\n')
for file in files_array:
    print("Checking scrapy results for "+file[5:-4]+" has completed...")
    out = os.popen('tail '+file+' -n 1')
    last = out.read()
    if("finished" in last):
        print("Scrapy crawl for "+file[5:-4]+" has completed...")
        print("Checking data completion...")
        data_file = os.popen('ls '+source+' | grep -E '+file[5:-4])
        data_filename = data_file.read().strip("\n")
        (from_date, to_date) = (data_filename.split('_')[2], data_filename.split('_')[4])
        date_data = os.popen('tail '+data_filename+' -c 200')
        new_from = re.search(r"[0-9]{4}-[0-9]{2}-[0-9]{2}", out.read()).group()
        print("Data for "+file[5:-4]+" to be scraped from "+from_date+" to "+to_date+" has scraped from "+new_from+" until "+to_date)
        r = re.compile('.*'+file+'.*')
        lines_matched = [line for line in queries if r.match(line)]
        query_match = lines_matched[0]
        queries.remove(query_match)
        done_query = (query_match.split(' ')[0] + query_match.split(' ')[1].split(':')[0] + ":" + new_from + query_match.split(' ')[2]).replace('*', '&')
        if(datetime.strptime(new_from, "%Y-%m-%d")-datetime.strptime(from_date, "%Y-%m-%d").months > 0):
            updated_query = (query_match.split(' ')[0] + query_match.split(' ')[1] + query_match.split(' ')[2].split(':')[1] + ":" + datetime.strftime(datetime.strptime(new_from, "%Y-%m-%d")-timedelta(days=1), "%Y-%m-%d")).replace('*', '?').replace('!', '$')
        os.popen('mv '+data_filename+' '+date_filename.replace(from_date, new_from))
        print("Renamed data file...")
        queries.extend([done_query, updated_query])
        print("Updated list of queries...")
        os.popen('mv /TweetScraper/Data/tweet/'+date_filename+' /data/')
        print("Moved data file...")
        os.open('rm '+log+'/'+file)
        print("Removed nohup file...")


with open('twitter_query_v2.txt', 'w') as f:
    for query in queries:
        f.write(query)
        f.write('\n')
f.close()

print("Queries have been updated!")
print("Key: ")
print("! Incumbent query")
print("$ Fixed date query")
print("* Running query")
print("$ Done query")
print("? Yet to run query")
