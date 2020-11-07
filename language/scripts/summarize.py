#!/usr/bin/env python3

from __future__ import print_function
import argparse, sys

def parse(text):
    return [int(x.replace('t: ', '')) for x in text.split('\n') if x.startswith('t: ')]

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description = 'Compute test duration')

    parser.add_argument(
        'files', nargs='+',
        help='Filenames to parse.')

    parser.add_argument(
        '--machine-readable', dest='machine_readable', action='store_true',
        help='Produce output in a machine readable format.')

    args = parser.parse_args()

    for path in args.files:
        if path == '-':
            t = parse(sys.stdin.read())
        else:
            with open(path, 'r') as f:
                t = parse(f.read())
        if not args.machine_readable:
            if len(t) == 0:
                print("%s: ERROR ERROR ERROR" % path)
            else:
                first = min(*t)
                last = max(*t)
                print("%s: %s %s %s" % (path, first/1e6, last/1e6, (last-first)/1e6))
        else:
            if len(t) == 0:
                print("ERROR" % path)
            else:
                first = min(*t)
                last = max(*t)
                print((last-first)/1e6)
