#!/bin/bash
files=`find /data/json/ -maxdepth 1 -name "#*.json"`
for file_name in $files 
do 
	echo $file_name
	temp_file=${file_name:10}
	new_file=${temp_file/json/csv}
	if [ ! -f /data/csv/$new_file ]; then
		jq -r '.[] | [.text] | @csv' $file_name > /data/csv/$new_file
	else
		echo "File already exists"
	fi
done
