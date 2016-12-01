#!/usr/bin/env python

from __future__ import print_function
import sys

if __name__ == '__main__':
    for path in sys.argv[1:]:
        with open(path, 'r') as f:
            t = [int(x.replace('t: ', '')) for x in f.read().split('\n') if x.startswith('t: ')]
        if len(t) == 0:
            print("%s: ERROR ERROR ERROR" % path)
        else:
            first = min(*t)
            last = max(*t)
            print("%s: %s %s %s" % (path, first/1e6, last/1e6, (last-first)/1e6))
