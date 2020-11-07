#!/usr/bin/env python3

import os
import sys
import argparse
import struct
import re
from operator import attrgetter

class CommaSplitAppend(argparse.Action):
    def __init__(self, option_strings, dest, nargs=None, **kwargs):
        if nargs is not None:
            raise ValueError("nargs not allowed")
        super(CommaSplitAppend, self).__init__(option_strings, dest, **kwargs)

    def __call__(self, parser, namespace, values, option_string=None):
        for s in values.split(','):
            a = getattr(namespace, self.dest)
            if a:
                a.append(s)
            else:
                setattr(namespace, self.dest, [s])

parser = argparse.ArgumentParser()
parser.add_argument('-l', '--list', action='store_true',
                    help='list gauges in file')
parser.add_argument('-i', '--include', action=CommaSplitAppend,
                    metavar='PATTERN',
                    help='regex(es) of gauges to print')
parser.add_argument('-x', '--exclude', action=CommaSplitAppend,
                    metavar='PATTERN',
                    help='regex(es) of gauges to NOT print')
parser.add_argument('-c', '--compress', action='store_true', dest='compress_unchanged',
                    help='compress sequences of identical samples')
parser.add_argument('-d', '--debug', action='store_true',
                    help='produce debugging output')
parser.add_argument('infile', help='name of input file (e.g. realmprof_0.dat)')
parser.add_argument('outfile', nargs='?',
                    help='name of output file (e.g. foo.csv)')

args = parser.parse_args(sys.argv[1:])

# packet and gauge types, hopefully consistent with runtime/realm/sampling.h

class PacketTypes:
    PACKET_EMPTY = 0
    PACKET_NEWGAUGE = 1
    PACKET_SAMPLES = 2

class GaugeTypes:
    GTYPE_UNKNOWN = 0
    GTYPE_ABSOLUTE = 1
    GTYPE_ABSOLUTERANGE = 2
    GTYPE_EVENTCOUNT = 3

gauges = dict()

column_names = { GaugeTypes.GTYPE_ABSOLUTE: ('value',),
                 GaugeTypes.GTYPE_ABSOLUTERANGE: ('value', 'min', 'max'),
                 GaugeTypes.GTYPE_EVENTCOUNT: ('count',)
}

class Gauge(object):
    def __init__(self, id, gtype, dtype, name):
        self.id = id
        self.gtype = gtype
        self.dtype = dtype.rstrip('\0')
        self.name = name.rstrip('\0')
        self.samples = {}
        self.total_samples = 0

        if gtype in (GaugeTypes.GTYPE_ABSOLUTE, GaugeTypes.GTYPE_EVENTCOUNT):
            if self.dtype == 'i':
                self.sample_size = 4
                self.sample_fmt = '<i'
            elif self.dtype == 'x':
                self.sample_size = 8
                self.sample_fmt = '<q'
            elif self.dtype == 'm':   # size_t
                self.sample_size = 8
                self.sample_fmt = '<Q'
            else:
                print 'unknown data type:', self.dtype
                assert False
        elif gtype == GaugeTypes.GTYPE_ABSOLUTERANGE:
            if self.dtype == 'i':
                self.sample_size = 12
                self.sample_fmt = '<iii'
            elif self.dtype == 'x':
                self.sample_size = 24
                self.sample_fmt = '<qqq'
            elif self.dtype == 'm':   # size_t
                self.sample_size = 24
                self.sample_fmt = '<QQQ'
            else:
                print 'unknown data type:', self.dtype
                assert False
        else:
            print 'unknown gauge type:', gtype
            assert False

    def add_samples(self, first_sample, last_sample, samples, runlengths):
        # uncompress samples
        if samples:
            u = []
            for s, l in zip(samples, runlengths):
                u.extend([ s for _ in xrange(l) ])
            samples = u
        self.samples[first_sample] = dict(first_sample = first_sample,
                                          last_sample = last_sample,
                                          samples = samples)
        self.total_samples += (last_sample - first_sample + 1)

    def print_info(self):
        print ' #{:3d}: {:<50s} ({:7d} samples)'.format(self.id,
                                                        self.name,
                                                        self.total_samples)

    def next_avail_sample(self, start_after):
        for first_sample in self.samples:
            if first_sample > start_after:
                return first_sample
            if self.samples[first_sample]['last_sample'] > start_after:
                return start_after + 1
        return None

    def get_sample(self, sample_index):
        for first_sample, s in self.samples.iteritems():
            if (first_sample <= sample_index) and (s['last_sample'] >= sample_index):
                return s['samples'][sample_index - first_sample]
        return None

with open(args.infile, 'rb') as f:
    while True:
        hdr = f.read(8)
        if len(hdr) < 8:
            break
        pkt_type, pkt_size = struct.unpack('<ii', hdr)

        if args.debug:
            print 'packet: type={:d} size={:d}'.format(pkt_type, pkt_size)

        if pkt_type == PacketTypes.PACKET_NEWGAUGE:
            pkt = f.read(64)
            assert len(pkt) == 64
            id, gtype, dtype, name = struct.unpack('<ii8s48s', pkt)

            # check to see if we want to show this gauge
            if args.include:
                if not(any(re.search(p, name) for p in args.include)):
                    continue
            if args.exclude:
                if any(re.search(p, name) for p in args.exclude):
                    continue

            g = Gauge(id, gtype, dtype, name)
            gauges[id] = g
            continue

        if pkt_type == PacketTypes.PACKET_SAMPLES:
            pkt = f.read(16)
            assert len(pkt) == 16
            id, comp_len, first_sample, last_sample = struct.unpack('<iiii', pkt)
            if id in gauges:
                g = gauges[id]
                assert pkt_size == (16 + comp_len * (g.sample_size + 2))
                if args.list:
                    f.seek(pkt_size - 16, 1)
                    g.add_samples(first_sample, last_sample, None, None)
                else:
                    sdata = f.read(comp_len * g.sample_size)
                    assert len(sdata) == (comp_len * g.sample_size)
                    rdata = f.read(comp_len * 2)
                    assert len(rdata) == (comp_len * 2)
                    samples = [ struct.unpack(g.sample_fmt, sdata[i * g.sample_size:(i + 1) * g.sample_size]) for i in xrange(comp_len) ]
                    runlengths = struct.unpack('<{:d}H'.format(comp_len), rdata)
                    g.add_samples(first_sample, last_sample, samples, runlengths)
            else:
                f.seek(pkt_size - 16, 1)
            continue

        # unrecognized packet type
        print 'unrecognized packet: type={:d} size={:d}'.format(pkt_type,
                                                                pkt_size)
        f.seek(pkt_size, 1) # relative to current position

if args.list:
    for g in sorted(gauges.values(), key=attrgetter('name')):
        g.print_info()
    exit(0)

# generate csv
if args.outfile:
    f = open(args.outfile, 'w')
else:
    f = sys.stdout

# header lines
hdr1 = [ 'sample' ]
hdr2 = [ 'index' ]

gs = sorted(gauges.values(), key=attrgetter('id'))
for g in gs:
    name = g.name
    for c in column_names[g.gtype]:
        hdr1.append('"{}"'.format(name) if (' ' in name) else name)
        name = ''
        hdr2.append(c)

f.write(','.join(hdr1) + '\n')
f.write(','.join(hdr2) + '\n')

sample = -1
prevrun = None
while True:
    sample = min(g.next_avail_sample(sample) for g in gs)
    if sample is None:
        break

    vs = []
    for g in gs:
        s = g.get_sample(sample)
        if s is not None:
            vs.extend([ str(v) for v in s ])
        else:
            vs.extend([ '' for c in column_names[g.gtype] ])
    l = ','.join(vs)
    if args.compress_unchanged:
        if prevrun and (sample == (prevrun[1]+1)) and (l == prevrun[2]):
            prevrun[1] = sample
        else:
            if prevrun:
                f.write('{:d},{}\n'.format(prevrun[0], prevrun[2]))
                f.write('{:d},{}\n'.format(prevrun[1], prevrun[2]))
            prevrun = [ sample, sample, l ]
    else:
        f.write('{:d},{}\n'.format(sample, l))
