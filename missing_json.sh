#!/bin/bash
file="$1"
file_size=$(wc -c <$file)
echo $file $file_size
temp_file=`echo $file | cut -d'/' -f 3`
start_year=`echo $file | cut -d'_' -f 3 | cut -d'-' -f 1`
end_year=`echo $file | cut -d'_' -f 5 | cut -d'-' -f 1`
start_year=$((start_year+1-1))
end_year=$((end_year+1-1))
for i in $(seq $start_year $end_year)
do
	json_size=$(wc -c<"/data/json/$temp_file?${i}.json")
	echo $json_size
	if [ -f "/data/json/$temp_file?${i}.json" -a $json_size -lt 10 ]; then
		echo $i
		months=("01" "02" "03" "04" "05" "06" "07" "08" "09" "10" "11" "12")
		for m in "${months[@]}"
		do
			jq '.' $file | jq -s ".[] | select((.datetime | contains(\"$i-$m\")) and .is_retweet==false and .is_reply==false) |  {query: .query, username: .usernameTweet, id: .ID, text: .text, nbr_retweet: .nbr_retweet, nbr_reply: .nbr_reply, nbr_favorite: .nbr_favorite, datetime: .datetime}" | jq -s . > "/data/json/$temp_file?${i}-${m}.json"
		done
	fi
done
