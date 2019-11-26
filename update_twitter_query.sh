#!/bin/bash
query=$1
file=`find ./Data/tweet/ -iname "#$query\_*"`
from=`echo $file | cut -d'_' -f3`
to=`echo $file | cut -d'_' -f5`
oldest=`tail $file -c 200 | grep -oE "[0-9]{4}-[0-9]{2}-[0-9]{2}"`
output=""
from="2019-10-15"
if [[ $from < $oldest ]]
then
  new_to=`date -d "$oldest - 1 days" +%Y-%m-%d`
  output="#$query since:$from until:$new_to"
fi
yesterday=`date -d "$date - 1 days" +%Y-%m-%d`
if [[ $to < $yesterday ]]
then
  new_from=`date -d "$to + 1 days" +%Y-%m-%d`
  if [[ -z $output ]]
  then
    output="#$query since:$new_from until:$yesterday"
  else
    output="$output;#$query since:$new_from until:$yesterday"
  fi
fi
echo $output
