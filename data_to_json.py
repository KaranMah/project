import os
import re
import json
import shlex
import subprocess

file_out = os.popen('ls . | grep -E "#.*"')
files = file_out.read()
files = files.strip('\n').split('\n')

for file_name in files:
    if(os.path.exists('/data/json/'+file_name+'.json')):
        print(file_name+" is already done...")
        pass
    else:
        print("Moving file "+file_name+" to /data/json folder...")
        subprocess.Popen(shlex.split("jq '.' "+file_name+" | jq -s '. | .[] | select(.is_retweet==false and .is_reply==false) |  {query: .query, username: .usernameTweet, id: .ID, text: .text, nbr_retweet: .nbr_retweet, nbr_reply: .nbr_reply, nbr_favorite: .nbr_favorite, datetime: .datetime}' | jq -s . > /data/json/"+file_name+".json &"))
        print("Moved...")
        print()

