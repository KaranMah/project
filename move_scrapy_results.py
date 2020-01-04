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

with open('twitter_query.txt', 'r') as f:
    queries = [x.strip('\n') for x in f.readlines()]

print("Log files located in "+log+"...")
print("Data files located in "+source+"...")
fileobj = os.popen('ls '+log+' | grep -E "nohup"')
files = fileobj.read()
files_array = files.strip('\n').split('\n')
for file_name in files_array:
    print("Checking scrapy results for "+file_name[5:-4]+" has completed...")
    out = os.popen('tail '+log+'/'+file_name+' -n 1')
    last = out.read()
    if("finished" in last):
        print("Scrapy crawl for "+file_name[5:-4]+" has completed...")
        print("Checking data completion...")
        try:
            data_file = os.popen('ls '+source+' | grep -E '+file_name[5:-4])
            data_filename = data_file.read().strip("\n")
            print("Reading file "+data_filename+"...")
            (from_date, to_date) = (data_filename.split('_')[2], data_filename.split('_')[4])
        except:
            continue
        date_data = os.popen('tail '+source+'/'+data_filename+' -c 150')
        new_from = re.search("[0-9]{4}-[0-9]{2}-[0-9]{2}", date_data.read()).group()
        print("Data for "+file_name[5:-4]+" to be scraped from "+from_date+" to "+to_date+" has scraped from "+new_from+" until "+to_date+"...")
        r = re.compile('\*.#'+file_name[5:-4]+' since:'+from_date+' until:'+to_date)
        lines_matched = [line for line in queries if r.match(line)]
        query_match = lines_matched[0]
        print("Query matched is ", query_match)
        queries.remove(query_match)
        done_query = '&'+ (query_match.split(' ')[0] + " since:" + new_from +" "+ query_match.split(' ')[2])[1:]
        if(from_date != new_from):
            os.popen('mv '+source+'/'+data_filename+' '+source+'/'+data_filename.replace(from_date, new_from))
            print("Renamed data file...")
        else:
            print("File name is the same...")
        date_diff = ((datetime.strptime(new_from, "%Y-%m-%d")-datetime.strptime(from_date, "%Y-%m-%d")).days)
        if(date_diff > 10):
            updated_query = '?' + ((query_match.split(' ')[0] +" "+ query_match.split(' ')[1] +" until:" + datetime.strftime(datetime.strptime(new_from, "%Y-%m-%d")-timedelta(days=1), "%Y-%m-%d")).replace('!', '$'))[1:]
            queries.append(updated_query)
        queries.append(done_query)
        print("Updated list of queries...")
        os.popen('mv '+source+'/'+data_filename.replace(from_date, new_from)+' /data/')
        print("Moved data file...")
        os.popen('rm '+log+'/'+file_name)
        print("Removed nohup file...")
        print()

with open('twitter_query.txt', 'w') as f:
    for query in queries:
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
print("**************************")
print()
