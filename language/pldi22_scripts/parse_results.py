#!/usr/bin/env python3

import csv
import glob
import os
import re

_filename_re = re.compile(r'out_([0-9]+)x([0-9]+)_r([0-9]+)[.]log')
def parse_basename(filename):
    match = re.match(_filename_re, filename)
    assert match is not None
    return match.groups()

_init_re = re.compile(r'^INIT TIME = +([0-9.]+) s$', re.MULTILINE)
_elapsed_re = re.compile(r'^ELAPSED TIME = +([0-9.]+) s$', re.MULTILINE)
def parse_content(path):
    with open(path, 'r') as f:
        content = f.read()
        init_match = re.search(_init_re, content)
        init = init_match.group(1) if init_match is not None else 'ERROR'
        elapsed_match = re.search(_elapsed_re, content)
        elapsed = elapsed_match.group(1) if elapsed_match is not None else 'ERROR'
        return (init, elapsed)

def main():
    paths = glob.glob('*/*.log')
    content = [(os.path.dirname(path),) + parse_basename(os.path.basename(path)) + parse_content(path) for path in paths]
    content.sort(key=lambda row: (row[0], int(row[1]), int(row[2]), int(row[3])))

    import sys
    # with open(out_filename, 'w') as f:
    out = csv.writer(sys.stdout, dialect='excel-tab') # f)
    out.writerow(['system', 'nodes', 'procs_per_node', 'rep', 'init_time', 'elapsed_time'])
    out.writerows(content)

if __name__ == '__main__':
    main()
