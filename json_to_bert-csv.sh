#!/bin/bash
files=`find . -maxdepth 1 -name "#*.json"`
for file in $files 
do
	echo $file
	jq -r '.[] | [.text] | @csv' $file > /data/csv/${file/.json/.csv}
done
