#!/bin/bash

for app in $(ls */ -d); do
  if [ "$app" == "compile_time/" ]; then continue; fi
  APP=`echo "##### $app #####" | sed s/"\/"/""/g`
  echo "$APP"
  for run in $(ls $app*/ -d); do
    RUN=`echo "$run" | cut -f2 -d'/'`
    echo "  * $RUN"
    PROBLEM_SIZE=`cat "$app"/problem_size`
    ITERATIONS=`cat "$app"/iterations`
    N=$(( PROBLEM_SIZE * ITERATIONS ))
    ./summarize.sh "$run" 2> /dev/null | awk -v n="$N" '{ print n / $1 }' | sed s/"^"/"    "/g
  done
done
