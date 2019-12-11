import re
import os
import sys
import shlex
import argparse
import subprocess
from psaw import PushshiftAPI
from datetime import datetime, timedelta

def sort_order(a, b):
    start_a = datetime.strptime(a.split('_')[1], "%d/%m/%Y")
    start_b = datetime.strptime(b.split('_')[1], "%d/%m/%Y")
    return start_a < start_b

if ((datetime.now()-timedelta(days=1) < datetime.strptime("31/12/2019", "%d/%m/%Y"))):
    yesterday = (datetime.now()-timedelta(days=1)).strftime("%d/%m/%Y")
else:
    yesterday = "31/12/2019"


parser = argparse.ArgumentParser(description='Script to run Reddit queries')
parser.add_argument('--queries', action="store", default="reddit_query.txt", dest="filename", help="Enter file name")
parser.add_argument('--log', action="store", default="Reddit_data/log/", dest="log", help="Enter log destination")
parser.add_argument('--data', action="store", default="Reddit_data/subreddit/", dest="data", help="Enter files destination")
parser.add_argument('--start', action="store", default="01/01/2010", dest="start", help="Enter start date dd/mm/yyyy")
parser.add_argument('--end', action="store", default=yesterday, dest="end", help="Enter end date dd/mm/yyyy")

log = parser.parse_args().log
data = parser.parse_args().data
file_name = parser.parse_args().filename
start_date = parser.parse_args().start
end_date = parser.parse_args().end


print("Log files located in "+log+"...")
print("Data files located in "+data+"...")

with open(file_name, 'r') as f:
    queries = [x.strip('\n') for x in f.readlines()]

while (True):
    query = queries.pop()
    queries.append(query)
    print("Checking if query is running")
    logfile_name = "nohup"+query+".out"
    runningobj = os.popen('ls '+log)
    files_obj = runningobj.read()
    files_array = files_obj.strip('\n').split('\n')
    print(files_array)
    if(logfile_name in files_array):
        print("The query is currently running...")
        continue
    else:
        print("The query is not currently running...")
        print("Checking if the query has been run before...")
        doneobj = os.popen('ls /data | grep -E r/'+query)
        file_obj = doneobj.read()
        done_queries = file_obj.strip('\n').split('\n')
        files = list(filter(lambda x: ("r/"+query) in x, done_queries))
        if(len(files) == 0):
            print("The query hasn't been run before...")
            os.system('nohup python3 scrape_reddit.py --query \"'+query+'\" --start \"'+start_date+'\" --end \"'+end_date+'\" > '+log+'nohup'+query+'.out &')
            print("Running query "+query+" from "+start_date+" till "+end_date)
            break
        else:
            print("The query has been run before...")
            print("Calculating remaining dates to run for...")
            sorted_files = sorted(files, key=sort_order)
            dates = list(filter(lambda x: (x.split('_')[1], x.split('_')[1]), sorted_files))
            (earliest_date, latest_date) = (dates[0][0], dates[-1][1])
            print("Dates collected previously are from "+earliest_date+" up to "+latest_date+"...")
            if(datetime.strptime(earliest_date, "%d/%m/%Y") > datetime.strptime(start_date, "%d/%m/%Y")):
                new_end = (datetime.strptime(earliest_date, "%d/%m/%Y")-timedelta(days=1)).strftime("%d/%m/%Y")
                os.system('nohup python scrape_reddit.py --query \"'+query+'\" --start \"'+start_date+'\" --end \"'+new_end+'\" > '+log+'nohup'+query+'.out &')
                print("Running query "+query+" from "+start_date+" till "+new_end)
                break
            elif(datetime.strptime(latest_date, "%d/%m/%Y") < datetime.strptime(yesterday, "%d/%m/%Y")):
                new_start = (datetime.strptime(latest_date, "%d/%m/%Y")+timedelta(days=1)).strftime("%d/%m/%Y")
                os.system('nohup python scrape_reddit.py --query \"'+query+'\" --start \"'+new_start+'\" --end \"'+end_date+'\" > '+log+'nohup'+query+'.out &')
                print("Running query "+query+" from "+new_start+" till "+end_date)
                break


with open(file_name, 'w') as f:
    for q in queries:
        f.write(q)
        f.write("\n")
