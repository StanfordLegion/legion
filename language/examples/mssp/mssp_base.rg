-- Copyright 2022 Stanford University
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
  exp_distance : float
}

fspace Edge(rn : region(Node)) {
  n1 : ptr(Node, rn),
  n2 : ptr(Node, rn),
  cost : float
}

task read_edge_data(g : GraphCfg, rn : region(Node), re : region(Edge(rn)))
  where reads writes(re.{n1,n2,cost})
do
  helpers.read_ptr_field(__runtime(), __context(), __physical(re.n1)[0], __fields(re.n1)[0],
			 g.datafile, 0)
  helpers.read_ptr_field(__runtime(), __context(), __physical(re.n2)[0], __fields(re.n2)[0],
			 g.datafile, g.edges * [ sizeof(int) ])

  helpers.read_float_field(__runtime(), __context(), __physical(re.cost)[0], __fields(re.cost)[0],
			   g.datafile, g.edges * [ sizeof(int) + sizeof(float) ])

  --for e in re do
  --  c.printf("%3d: %3d %3d %5.3f\n", __raw(e).value, __raw(e.n1).value, __raw(e.n2).value, e.cost)
  --end
end

task sssp(g : GraphCfg, rn : region(Node), re : region(Edge(rn)), root : ptr(Node, rn))
  where reads(re.{n1,n2,cost}), reads writes(rn.{distance})
do
  -- fill called by parent
  --fill(rn.distance, INFINITY)
  root.distance = 0

  -- upper bound is |V|-1 iterations - should normally take much less than that
  for steps = 1, g.nodes do
    var count = 0
    for e in re do
      var d1 = e.n1.distance
      var d2 = e.n2.distance
      if (d1 + e.cost) < d2 then
	e.n2.distance = d1 + e.cost
	count = count + 1
      end
    end
    if count == 0 then
      break
    end
  end
  return 0
end

task read_expected_distances(g : GraphCfg, rn : region(Node), filename : &int8)
  where reads writes(rn.exp_distance)
do
  helpers.read_float_field(__runtime(), __context(),
         __physical(rn.exp_distance)[0],
         __fields(rn.exp_distance)[0],
			   filename, 0)
end

task check_results(g : GraphCfg, rn : region(Node), verbose : bool)
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
  do
    var i = 1
    while i < args.argc do
      if cstring.strcmp(args.argv[i], '-v') == 0 then
	verbose = true
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
  var re = region(ispace(ptr, graph.edges), Edge(rn))

  read_edge_data(graph, rn, re)

  for s = 0, graph.num_sources do
    fill(rn.distance, INFINITY)

    var root_id = graph.sources[s]
    var root : ptr(Node, rn) = dynamic_cast(ptr(Node, rn), ptr(root_id))
    var ts_start = c.legion_get_current_time_in_micros()
    wait_for(sssp(graph, rn, re, root))
    var ts_end = c.legion_get_current_time_in_micros()

    read_expected_distances(graph, rn, [&int8](graph.expecteds[s]))
    var errors = check_results(graph, rn, verbose)
    if errors == 0 then
      c.printf("source %d OK, elapsed time: %.3f ms\n", root_id, (ts_end - ts_start) * 1e-3)
    else
      c.printf("source %d - %d errors!\n", root_id, errors)
    end
  end
end

regentlib.start(toplevel)
