#!/bin/bash
files=`find /data/json/ -maxdepth 1 -name "#Australia_since_2014-06*.json"`
for file in $files 
do
	echo $file
	temp_file=`echo $file | cut -d'/' -f 3`
	jq -r '.[] | [.text] | @csv' $file > /data/csv/${temp_file/.json/.csv}
done
