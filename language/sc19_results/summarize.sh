#!/bin/bash

[ $# -gt 0 ] || (echo "Usage: $0 (directory)"; exit 1;)

for i in 1 2 4 8 16 32 64 128 256; do
  grep ELAPSED "$1"/out_"$i"x1* | awk 'BEGIN { cnt = 0; sum = 0; } {cnt +=1; sum += $4} END { print sum / cnt; }'
done
