#!/usr/bin/env python

from __future__ import print_function
import sys
import numpy

print('%e' % numpy.mean([float(x) for x in sys.stdin.read().split()]))
