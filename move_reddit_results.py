import re
import os
import sys
import json
import shlex
import argparse
import subprocess
from datetime import datetime, timedelta


parser = argparse.ArgumentParser(description='Script to run Reddit queries')
parser.add_argument('--log', action="store", default="Reddit_data/log/", dest="log", help="Enter log destination")
parser.add_argument('--data', action="store", default="Reddit_data/subreddit/", dest="data", help="Enter files destination")

log = parser.parse_args().log
data = parser.parse_args().data


print("Log files located in "+log+"...")
print("Data files located in "+data+"...")

fileobj = os.popen('ls '+log+' | grep -E "nohup"')
files = fileobj.read()
files_array = files.strip('\n').split('\n')
nohup_files = list(filter(lambda x: "nohup" in x, files_array))
for file_name in nohup_files:
    query = file_name[5:-4]
    if(len(query) > 0):
        print("Subreddit: "+query)
        print("Checking if scraping results for "+query+" are completed...")
        out = os.popen('tail '+log+'/'+file_name+' -n 1')
        last = out.read()
        if("Completed" in last):
            print("Scraping is completed")
            print("Locating data file for "+query+"...")
            if(os.path.exists(data+query)):
                print("Data file exists for "+query+"...")
                from_date = "01-01-2010"
                to_date = "01-01-2019"
                file_start = os.popen('head '+data+query+' -n 1 | tail -c 20 | grep -Eo "[0-9]{10}"')
                first_line = file_start.read()
                from_date = (datetime.strftime(datetime.fromtimestamp(float(first_line)), "%d-%m-%Y"))
                file_end = os.popen('tail '+data+query+' -n 1 | tail -c 20 | grep -Eo "[0-9]{10}"')
                last_line = file_end.read()
                to_date = (datetime.strftime(datetime.fromtimestamp(float(last_line)), "%d-%m-%Y"))
                print("Data file for "+query+" has scraped from "+from_date+" to "+to_date)
                perm_file = 'r_'+query+'_'+from_date+'_'+to_date
                os.popen('mv '+data+query+' /data/'+perm_file)
                print("Moved data file...")
                os.popen('rm '+log+file_name)
                print("Removed nohup file...")
                print()
