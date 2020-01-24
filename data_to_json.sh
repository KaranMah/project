#!/bin/bash
files=`find /data/ -maxdepth 1 -name "#*"`
echo $files
for file in $files 
do
	echo $file
	temp_file=`echo $file | cut -d'/' -f 3`
	start_year=`echo $file | cut -d'_' -f 3 | cut -d'-' -f 1`
	end_year=`echo $file | cut -d'_' -f 5 | cut -d'-' -f 1`
	start_year=$((start_year+1-1))
	end_year=$((end_year+1-1))
	for i in $(seq $start_year $end_year)
	do
		if [ ! -f "/data/json/$temp_file?${i}.json" ]; then
			echo $i
			jq '.' $file | jq -s ".[] | select((.datetime | contains(\"$i\")) and .is_retweet==false and .is_reply==false) |  {query: .query, username: .usernameTweet, id: .ID, text: .text, nbr_retweet: .nbr_retweet, nbr_reply: .nbr_reply, nbr_favorite: .nbr_favorite, datetime: .datetime}" | jq -s . > "/data/json/$temp_file?${i}.json"
		fi
	done
done
