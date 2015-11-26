#!/usr/bin/env python

import os
import sys
import getopt
import random
import array

opts, args = getopt.getopt(sys.argv[1:], 'n:e:s:c:p:o:r:v')
opts = dict(opts)

nodes = int(opts.get('-n', '10'))
edges = int(opts.get('-e', '20'))
subgraphs = int(opts.get('-s', '1'))
cluster_factor = int(opts.get('-c', '95'))
problems = int(opts.get('-p', '1'))
randseed = int(opts.get('-r', '12345'))
outdir = opts['-o']
verbose = '-v' in opts

random.seed(randseed)

def create_graph(nodes, edges):
    n1 = [ random.randint(0, nodes - 1) for x in xrange(edges) ]
    n2 = [ random.randint(0, nodes - 1) for x in xrange(edges) ]
    length = [ random.expovariate(1.0) for x in xrange(edges) ]
    return { 'nodes': nodes,
             'edges': edges,
             'n1': n1,
             'n2': n2,
             'length': length }

def solve_graph(g, source):
    parent = [ -1 for x in xrange(g['nodes']) ]
    dist = [ 1e100 for x in xrange(g['nodes']) ]
    dist[source] = 0
    while True:
        count = 0
        for n1, n2, length in zip(g['n1'], g['n2'], g['length']):
            c2 = length + dist[n1]
            if c2 < dist[n2]:
                dist[n2] = c2
                parent[n2] = n1
                count += 1

        #print 'count = {:d}'.format(count)
        if count == 0:
            break

    if verbose:
        for i, e in enumerate(zip(g['n1'], g['n2'], g['length'])):
            print '{:3d} {:3d} {:3d} {:5.3f}'.format(i, e[0], e[1], e[2])
        for i, n in enumerate(zip(parent, dist)):
            print '{:3d} {:3d} {:5.3f}'.format(i, n[0], n[1])

    return dist

G = create_graph(nodes, edges)

if not os.path.exists(outdir):
    os.mkdir(outdir)
else:
    assert os.path.isdir(outdir)

with open(os.path.join(outdir, 'edges.dat'), 'wb') as f:
    array.array('i', G['n1']).tofile(f)
    array.array('i', G['n2']).tofile(f)
    array.array('f', G['length']).tofile(f)

with open(os.path.join(outdir, 'graph.txt'), 'w') as f:
    f.write('nodes {:d}\n'.format(nodes))
    f.write('edges {:d}\n'.format(edges))
    f.write('data edges.dat\n')

    sources = random.sample(xrange(nodes), problems)
    for s in sources:
        parents = solve_graph(G, s)
        with open(os.path.join(outdir, 'result_{:d}.dat'.format(s)), 'wb') as f2:
            array.array('f', parents).tofile(f2)

        f.write('source {:d} result_{:d}.dat\n'.format(s, s))

