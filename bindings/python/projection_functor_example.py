from __future__ import print_function

import pygion
from pygion import index_launch, task, Domain, ID, IndexLaunch, R, Region, Partition, ProjectionFunctor

from typing import cast, Callable

import subprocess
import petra as pt

f = ProjectionFunctor(1 + 2*ID)

@task(privileges=[R])
def hello(R, i):
    print("hello from point %s (region %s)" % (i, R.ispace.bounds))

@task
def main():
    R = Region([10], {"x": pygion.float64})
    P = Partition.equal(R, [4])
    for i in range(4):
        print("python region %s is %s %s %s" % (i, P[i].handle[0].tree_id, P[i].handle[0].index_space.tid, P[i].handle[0].index_space.id))
    pygion.fill(R, "x", 0)

    for i in IndexLaunch([3]):
        # hello(P[f(i)], i)
        # add +1 here after preprocess
        # if not only i contruct it then use it 
        # its going to be inneficient 
        # dictionary key is expression anf val is proj function
        # have our own hash function
        hello(P[i+1], i)
        # step 1:
        # change this to Pfi instead have P[i+1]
        # have to go back to argcheck and detect math expressions and turn it into proj functor on the fly
        # step2:
        # improve efficiency

    # have one like line 65 in index_launch.py
    index_launch([3], hello, P[ID], ID)


if __name__ == "__main__":
    main()

# symbolicexpr in line 2264 in link