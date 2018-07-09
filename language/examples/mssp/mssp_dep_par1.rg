-- Copyright 2018 Stanford University
--
-- Licensed under the Apache License, Version 2.0 (the "License");
-- you may not use this file except in compliance with the License.
-- You may obtain a copy of the License at
--
--     http://www.apache.org/licenses/LICENSE-2.0
--
-- Unless required by applicable law or agreed to in writing, software
-- distributed under the License is distributed on an "AS IS" BASIS,
-- WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
-- See the License for the specific language governing permissions and
-- limitations under the License.

-- This sample program implements an MSSP (Multi-Source Shortest Path)
-- analysis on an arbitrary directed graph using the Bellman-Ford algorithm
-- (i.e. the same one used in the standard "SSSP" formulation).
--
-- The graph topology is defined by Edge's, which are directed edges from 'n1'
-- to 'n2' with a possibly-negative cost.  The data is in a file that can be
-- read in (in a distributed manner, if you like) with some helper functions.
-- The graph description also includes one or more source node IDs, which are
-- the origins of the individual SSSP problems, which may be processed serially,
-- or in parallel.
--
-- This version accept an additional -p N option that divides the nodes and edges
-- into N ~equal subregions and parallelizes the loading, analysis, and checking.
-- No attempt is made to line up the nodes and edges partitions though, so the
-- the update step for a subset of edges still requires access to all the nodes,
-- preventing us from doing any local iteration.

-- This is for the automated Regent test suite:
-- runs-with:
-- [ ["small"] ]

import "regent"

-- these give us useful things like c.printf, c.exit, cstring.strcmp, ...
local c = regentlib.c
local cstring = terralib.includec("string.h")
local cmath = terralib.includec("math.h")
local INFINITY = 1.0/0.0

local GraphCfg = require("mssp_graphcfg")

local helpers = require("mssp_helpers")

fspace Node {
  distance : float,
  dist_next : float,
  exp_distance : float,
}

fspace Edge(rsrc : region(Node), rdst : region(Node)) {
  n1 : ptr(Node, rsrc),
  n2 : ptr(Node, rdst),
  cost : float,
}

task read_edge_data(g : GraphCfg, re : region(Edge(wild, wild)))
  where reads writes(re.{n1,n2,cost})
do
  helpers.read_ptr_field(__runtime(), __context(), __physical(re)[0], __fields(re)[0],
			 g.datafile, 0)
  helpers.read_ptr_field(__runtime(), __context(), __physical(re)[1], __fields(re)[1],
			 g.datafile, g.edges * [ sizeof(int) ])

  helpers.read_float_field(__runtime(), __context(), __physical(re)[2], __fields(re)[2],
			   g.datafile, g.edges * [ sizeof(int) + sizeof(float) ])

  --for e in re do
  --  c.printf("%3d: %3d %3d %5.3f\n", __raw(e).value, __raw(e.n1).value, __raw(e.n2).value, e.cost)
  --end
end

task sssp_update(rsrc : region(Node), rdst : region(Node),
                 re : region(Edge(rsrc, rdst)))
  where reads(re.{n1,n2,cost}), reads(rsrc.distance),
        reduces min(rdst.dist_next)
do
  for e in re do
    var d1 = e.n1.distance
    e.n2.dist_next min= d1 + e.cost
  end
end

task sssp_collect(rn : region(Node))
  where reads writes(rn.distance), reads(rn.dist_next)
do
  var count = 0
  for n in rn do
    var oldval = n.distance
    var newval = n.dist_next
    if (newval < oldval) then
      n.distance = newval
      count = count + 1
    end
  end
  return count
end

task sssp(g : GraphCfg, subgraphs : int,
	  rn : region(Node), re : region(Edge(wild, wild)),
	  psrc : partition(disjoint, rn, ispace(int1d)),
	  pdst : partition(aliased, rn, ispace(int1d)),
          pe : partition(disjoint, re, ispace(int1d)),
	  root : ptr(Node, rn))
  where reads(re.{n1,n2,cost}), reads writes(rn.{distance,dist_next})
do
  -- fill called by parent
  --fill(rn.distance, INFINITY)
  root.distance = 0

  -- upper bound is |V|-1 iterations - should normally take much less than that
  for steps = 1, g.nodes do
    c.legion_runtime_begin_trace(__runtime(), __context(), 0, false)

    var count = 0
    __demand(__parallel)
    for i = 0, subgraphs do
      sssp_update(psrc[i], pdst[i], pe[i])
    end

    __demand(__parallel)
    for i = 0, subgraphs do
      count += sssp_collect(psrc[i])
    end

    c.legion_runtime_end_trace(__runtime(), __context(), 0)
    if count == 0 then
      break
    end
  end
  return 0
end

task read_expected_distances(rn : region(Node), filename : &int8)
  where reads writes(rn.exp_distance)
do
  helpers.read_float_field(__runtime(), __context(), __physical(rn)[0], __fields(rn)[0],
			   filename, 0)
end

task check_results(rn : region(Node), verbose : bool)
  where reads(rn.{distance,exp_distance})
do
  var errors = 0
  for n in rn do
    var d = n.distance
    var ed = n.exp_distance
    if (d == ed) or (cmath.fabsf(d - ed) < 1e-5) then
      -- ok
    else
      if verbose then
	c.printf("MISMATCH on node %d: parent=%5.3f expected=%5.3f\n", __raw(n).value, d, ed)
      end
      errors = errors + 1
    end
  end
  return errors
end

terra wait_for(x : int)
  return x
end

task toplevel()
  -- ask the Legion runtime for our command line arguments
  var args = c.legion_runtime_get_input_args()

  var graph : GraphCfg
  var verbose = false
  var subgraphs = 1
  do
    var i = 1
    while i < args.argc do
      if cstring.strcmp(args.argv[i], '-v') == 0 then
	verbose = true
      elseif cstring.strcmp(args.argv[i], '-p') == 0 then
	i = i + 1
	subgraphs = c.atoi(args.argv[i])
      else
	break
      end
      i = i + 1
    end
    if i >= args.argc then
      c.printf("Usage: %s [-v] cfgdir\n", args.argv[0])
      c.exit(1)
    end
    if verbose then
      c.printf("reading config from '%s'...\n", args.argv[i])
    end
    graph:read_config_file(args.argv[i])
  end
  graph:show()

  var rn = region(ispace(ptr, graph.nodes), Node)
  var re = region(ispace(ptr, graph.edges), Edge(rn, rn))

  var colors = ispace(int1d, subgraphs)
  var psrc = partition(equal, rn, colors)

  -- equal subspaces of re
  var pe_disjoint = partition(equal, re, colors)

  -- parallel load of edge data
  __demand(__parallel)
  for i = 0, subgraphs do
    read_edge_data(graph, pe_disjoint[i])
  end

  -- preimage partition of re
  var pe = preimage(re, psrc, re.n1)
  var pdst = image(rn, pe, re.n2)

  for s = 0, graph.num_sources do
    fill(rn.distance, INFINITY)
    fill(rn.dist_next, INFINITY)

    var root_id = graph.sources[s]
    var root : ptr(Node, rn) = dynamic_cast(ptr(Node, rn), [ptr](root_id))
    var ts_start = c.legion_get_current_time_in_micros()
    wait_for(sssp(graph, subgraphs, rn, re, psrc, pdst, pe, root))
    var ts_end = c.legion_get_current_time_in_micros()

    for i = 0, subgraphs do
      read_expected_distances(psrc[i], [&int8](graph.expecteds[s]))
    end
    var errors = 0
    for i = 0, subgraphs do
      errors += check_results(psrc[i], verbose)
    end
    if errors == 0 then
      c.printf("source %d OK, elapsed time: %.3f ms\n", root_id, (ts_end - ts_start) * 1e-3)
    else
      c.printf("source %d - %d errors!\n", root_id, errors)
    end
  end
end

regentlib.start(toplevel)
