#!/usr/bin/env python

from __future__ import print_function
import collections
import sys

if __name__ == '__main__':
    for path in sys.argv[1:]:
        with open(path, 'r') as f:
            t = collections.defaultdict(list)
            for line in f:
                if line.startswith('t: '):
                    proc, time = map(int, line.replace('t: ', '').split())
                    t[proc].append(time)
        if len(t) == 0:
            print("%s: ERROR" % path)
        else:
            total = max(*(max(*x) - min(*x) for x in t.values()))
            print("%s: %s" % (path, total/1e6))
