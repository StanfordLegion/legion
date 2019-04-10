#!/bin/bash

for dir in $(ls */ -d); do
  APP=`echo "##### $dir #####" | sed s/"\/"/""/g`
  echo "$APP"
  for run in $(ls $dir/); do
    echo "  * $run"
    ./summarize.sh "$dir/$run" 2> /dev/null | sed s/"^"/"    "/g
  done
done
