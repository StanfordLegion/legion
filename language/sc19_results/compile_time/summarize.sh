#!/bin/bash

[ $# -gt 0 ] || (echo "Usage: $0 (directory)"; exit 1;)

TASKS=`grep [.]parallel "$1"/compile_auto | grep Starting | awk '{print $6}' | sed s/"[.]parallel"/""/g | sort | uniq`
TIME_INFER=`for t in "$TASKS"; do grep "$t" "$1"/compile_auto | grep parallelize_tasks | awk '{print $5}'; done | sum.rb`
TIME_SOLVE=`grep solve "$1"/compile_auto | head -1 | awk '{print $5}'`
TIME_REWRITE=`grep rewrite "$1"/compile_auto | head -1 | awk '{print $5}'`
TIME_TOTAL=`grep elapsed "$1"/compile_auto | awk '{print $1}' | sed s/"user"//g`
TIME_TOTAL_MANUAL=`grep elapsed "$1"/compile_manual | awk '{print $1}' | sed s/"user"//g`
TIME_CODEGEN=`ruby -e "puts $TIME_TOTAL - $TIME_INFER - $TIME_SOLVE - $TIME_REWRITE"`

echo "$TIME_INFER" | awk '{printf "inference: %.1fms\n", $1*1000}'
echo "$TIME_SOLVE" | awk '{printf "solve: %.1fms\n", $1*1000}'
echo "$TIME_REWRITE""s" | awk '{printf "rewrite: %.1fs\n", $1}'
echo "$TIME_CODEGEN""s" | awk '{printf "codegen: %.1fs\n", $1}'
echo "$TIME_TOTAL""s" | awk '{printf "total: %.1fs\n", $1}'
echo "$TIME_TOTAL_MANUAL""s" | awk '{printf "total (manual): %.1fs\n", $1}'
