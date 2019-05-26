#!/bin/bash

for app in $(ls */ -d); do
  echo "##### $app #####" | sed s/"\/"/""/g
  ./summarize.sh "$app"
done
