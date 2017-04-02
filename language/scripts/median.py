#!/usr/bin/env python

from __future__ import print_function
import sys
import numpy

print('%e' % numpy.median([float(x) for x in sys.stdin.read().split()]))
