from __future__ import print_function

import pygion
from pygion import (
    index_launch,
    task,
    Domain,
    ID,
    IndexLaunch,
    R,
    Region,
    Partition,
    ProjectionFunctor,
)

from typing import cast, Callable

import subprocess
import petra as pt

f = ProjectionFunctor.create(1 + ID)


@task(privileges=[R])
def hello(R, i, num):
    print("hello from point %s (region %s)" % (i, R.ispace.bounds))
    assert int(R.ispace.bounds[0, 0]) == int(i + num)


@task
def main():
    R = Region([4], {"x": pygion.float64})
    P = Partition.equal(R, [4])
    for i in range(4):
        print(
            "python region %s is %s %s %s"
            % (
                i,
                P[i].handle[0].tree_id,
                P[i].handle[0].index_space.tid,
                P[i].handle[0].index_space.id,
            )
        )
    pygion.fill(R, "x", 0)

    for i in IndexLaunch([3]):
        hello(P[f(i)], i, 1)

    for i in IndexLaunch([3]):
        hello(P[i], i, 0)

    for i in IndexLaunch([2]):
        hello(P[i + 2], i, 2)

    for i in IndexLaunch([2]):
        hello(P[i + 2], i, 2)

    index_launch([3], hello, P[ID], ID, 0)

    # This Seg Fault when running all tests but not when it's the only test:
    # index_launch([3], hello, P[f(ID)], ID, 1)

    index_launch([2], hello, P[ID + 2], ID, 2)


if __name__ == "__main__":
    main()
