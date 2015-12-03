#!/usr/bin/env python

import argparse
import array
import getopt
import os
import random
import sys
import subprocess

def create_graph(nodes, edges):
    n1 = [ random.randint(0, nodes - 1) for x in xrange(edges) ]
    n2 = [ random.randint(0, nodes - 1) for x in xrange(edges) ]
    length = [ random.expovariate(1.0) for x in xrange(edges) ]
    return { 'nodes': nodes,
             'edges': edges,
             'n1': n1,
             'n2': n2,
             'length': length }

def metis_graph(g, metis, subgraphs, outdir):
    with open(os.path.join(outdir, 'graph.metis'), 'wb') as f:
        f.write('{:3d} {:3d} 000\n'.format(g['nodes'], g['edges']))
        for n in xrange(g['nodes']):
            f.write(' '.join('{:3d} 1'.format(n2+1) for n1, n2 in zip(g['n1'], g['n2']) if n1 == n))
            f.write('\n')
    subprocess.check_call([metis, os.path.join(outdir, 'graph.metis'), str(subgraphs)])
    with open(os.path.join(outdir, 'graph.metis.part.{}'.format(subgraphs)), 'rb') as f:
        colors = [int(x) - 1 for x in f.read().split()]
    mapping = dict(zip(sorted(range(g['nodes']), key = lambda x: colors[x]), range(g['nodes'])))
    g['n1'] = [mapping[g['n1'][x]] for x in xrange(g['edges'])]
    g['n2'] = [mapping[g['n2'][x]] for x in xrange(g['edges'])]

def solve_graph(g, source, verbose):
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

if __name__ == '__main__':
    p = argparse.ArgumentParser(description = 'graph generator')
    p.add_argument('--nodes', '-n', type = int, default = 10)
    p.add_argument('--edges', '-e', type = int, default = 20)
    p.add_argument('--subgraphs', '-s', type = int, default = 1)
    p.add_argument('--cluster-factor', '-c', type = int, default = 95)
    p.add_argument('--problems', '-p', type = int, default = 1)
    p.add_argument('--randseed', '-r', type = int, default = 12345)
    p.add_argument('--metis', '-m', default = './metis-install/bin/gpmetis')
    p.add_argument('--outdir', '-o', required = True)
    p.add_argument('--verbose', '-v', action = 'store_true')
    args = p.parse_args()

    random.seed(args.randseed)

    G = create_graph(args.nodes, args.edges)

    try:
        os.mkdir(args.outdir)
    except:
        pass
    assert os.path.isdir(args.outdir)

    if len(args.metis) > 0:
        assert os.path.isfile(args.metis)
        metis_graph(G, args.metis, args.subgraphs, args.outdir)

    with open(os.path.join(args.outdir, 'edges.dat'), 'wb') as f:
        array.array('i', G['n1']).tofile(f)
        array.array('i', G['n2']).tofile(f)
        array.array('f', G['length']).tofile(f)

    with open(os.path.join(args.outdir, 'graph.txt'), 'w') as f:
        f.write('nodes {:d}\n'.format(args.nodes))
        f.write('edges {:d}\n'.format(args.edges))
        f.write('data edges.dat\n')

        sources = random.sample(xrange(args.nodes), args.problems)
        for s in sources:
            parents = solve_graph(G, s, args.verbose)
            with open(os.path.join(args.outdir, 'result_{:d}.dat'.format(s)), 'wb') as f2:
                array.array('f', parents).tofile(f2)

            f.write('source {:d} result_{:d}.dat\n'.format(s, s))
