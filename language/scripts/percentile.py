#!/usr/bin/env python

from __future__ import print_function
import sys
import numpy

a = int(sys.argv[1])
print('%e' % numpy.percentile([float(x) for x in sys.stdin.read().split()], a))
