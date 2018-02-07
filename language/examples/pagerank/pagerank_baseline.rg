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

-- runs-with:
-- []

import "regent"
import "bishop"

mapper

$GPUs = processors[isa=cuda]
$CPUs = processors[isa=x86]
$HAS_GPUS = $GPUs.size > 0

task#pagerank[index=$i] {
  target : $HAS_GPUS ? $GPUs[$i % $GPUs.size] : $CPUs[$i % $CPUs.size];
}

task#init_graph[index=$i],
task#init_pr_score[index=$i],
task#init_partition[index=$i] {
  target : $CPUs[$i % $CPUs.size];
}

task#pagerank[target=$proc] region#pr_old,
task#pagerank[target=$proc] region#pr_new,
task#pagerank[target=$proc] region#nodes,
task#pagerank[target=$proc] region#edges {
  target : $HAS_GPUS ? $proc.memories[kind=fbmem]
                     : $proc.memories[kind=sysmem];
}

task#init_pr_score[target=$proc] region#pr,
task#init_graph[target=$proc] region#nodes,
task#init_graph[target=$proc] region#edges,
task#init_partition[target=$proc] region#edge_range,
task#init_partition[target=$proc] region#edges {
  target : $HAS_GPUS ? $proc.memories[kind=zcmem]
                     : $proc.memories[kind=sysmem];
}

end

local c = regentlib.c
local std = terralib.includec("stdlib.h")
local cstring = terralib.includec("string.h")
local V_ID = int32
local E_ID = int64

struct Config {
  num_nodes : V_ID,
  num_edges : E_ID,
  num_iterations : int32,
  num_workers : int32
  graph : int8[128]
}

fspace NodeStruct {
  index : E_ID,
  degree : V_ID
}

fspace EdgeStruct {
  src : V_ID,
  dst : int1d
}

terra parse_input_args(conf : Config)
  var args = c.legion_runtime_get_input_args()
  var input_file : rawstring
  for i = 0, args.argc do
    if cstring.strcmp(args.argv[i], "-ni") == 0 then
      i = i + 1
      conf.num_iterations = std.atoi(args.argv[i])
    elseif cstring.strcmp(args.argv[i], "-nw") == 0 then
      i = i + 1
      conf.num_workers = std.atoi(args.argv[i])
    elseif cstring.strcmp(args.argv[i], "-graph") == 0 then
      i = i + 1
      input_file = rawstring(args.argv[i])
    end
  end
  var file = c.fopen(input_file, "r")
  c.fscanf(file, "%i", &conf.num_nodes)
  c.fscanf(file, "%i", &conf.num_edges)
  c.fscanf(file, "%s", conf.graph)
  c.fclose(file)
  return conf
end

task init_graph(nodes : region(ispace(int1d), NodeStruct),
                edges : region(ispace(int1d), EdgeStruct),
                num_nodes : V_ID,
                num_edges : E_ID,
                graph : int8[128])
where
  reads(nodes, edges), writes(nodes, edges)
do
  var indices : &E_ID = [&E_ID](c.malloc(num_nodes * 8))
  var degrees : &V_ID = [&V_ID](c.malloc(num_nodes * 4))
  var srcs : &V_ID = [&V_ID](c.malloc(num_edges * 4))
  var file = c.fopen(graph, "rb")
  c.printf("graph = %s\n", graph)
  c.fread(indices, 8, num_nodes, file)
  for n = 0, num_nodes do
    nodes[n].index = indices[n]
  end
  c.fread(degrees, 4, num_nodes, file)
  for n = 0, num_nodes do
    nodes[n].degree = degrees[n]
  end
  c.fread(srcs, 4, num_edges, file)
  var dst : V_ID = 0
  for e = 0, num_edges do
    while nodes[dst].index <= e do
      dst = dst + 1;
    end
    edges[e].src = srcs[e]
    -- The dst field holds an index to node region, which is
    -- used for dependent partitioning
    edges[e].dst = dynamic_cast(int1d(NodeStruct, nodes), dst)
  end
  c.fclose(file)
  c.free(indices)
  c.free(degrees)
  c.free(srcs)
  
  return 1
end

task init_pr_score(pr : region(ispace(int1d), float),
                   num_nodes : int64)
where
  writes(pr)
do
  var init_score = 1.0f / num_nodes
  c.printf("init_score = %.8lf\n", init_score)
  for n in pr do
    pr[n] = init_score
  end
  return 1
end

task init_partition(edge_range : region(ispace(int1d), regentlib.rect1d),
                    edges : region(ispace(int1d), EdgeStruct),
                    avg_num_edges : E_ID,
                    num_parts : int)
where
  writes(edge_range), reads(edges)
do
  var left_bound : E_ID = 0
  var right_bound : E_ID = 0
  var total_num_edges : E_ID = edges.ispace.bounds.hi
  for p = 0, num_parts do
    right_bound = min(avg_num_edges * (p + 1), total_num_edges)
    var my_dst   : V_ID = edges[right_bound].dst
    -- extend the right bound of this subregion until we find
    -- the last edge of the current node
    while (right_bound < total_num_edges) do
      var next_dst : V_ID = edges[right_bound+1].dst
      if (my_dst < next_dst) then
        break
      end
      right_bound = right_bound + 1
    end
    edge_range[p] = {left_bound, right_bound}
    left_bound = right_bound + 1
  end
  return left_bound
end

__demand(__cuda)
task pagerank(nodes : region(ispace(int1d), NodeStruct),
              edges : region(ispace(int1d), EdgeStruct),
              pr_old : region(ispace(int1d), float),
              pr_new : region(ispace(int1d), float),
              alpha : float, num_nodes : V_ID)
where
  reads(nodes, edges, pr_old), writes(pr_new)
do
  var node_is = nodes.ispace
  var edge_is = edges.ispace
  for n in node_is do
    var left = 0
    if n > node_is.bounds.lo then
      left = nodes[n-1].index
    else
      left = edge_is.bounds.lo
    end
    var right = nodes[n].index
    var score = 0.0f
    for e = left, right do
      score = score + pr_old[edges[e].src]
    end
    var out_degree = nodes[n].degree
    -- avoid divided-by-zero errors
    if out_degree == 0 then
      out_degree = 1
    end
    score = ((1 - alpha) / num_nodes + alpha * score)
    pr_new[n] = score / out_degree
  end
end

task main()
  var conf : Config
  conf.num_nodes = 10000
  conf.num_edges = 1000000
  conf.num_iterations = 10
  conf.num_workers = 1
  conf = parse_input_args(conf)
  c.printf("pagerank settings: num_nodes=%d num_edges=%lld iterations=%d workers=%d\n",
            conf.num_nodes, conf.num_edges, conf.num_iterations, conf.num_workers)
  var is_nodes = ispace(int1d, conf.num_nodes)
  var is_edges = ispace(int1d, conf.num_edges)

  var all_nodes = region(is_nodes, NodeStruct)
  var all_edges = region(is_edges, EdgeStruct)

  var pr_score0 = region(is_nodes, float)
  var pr_score1 = region(is_nodes, float)

  c.printf("Load input graph...\n")
  init_graph(all_nodes, all_edges, conf.num_nodes, conf.num_edges, conf.graph)
  init_pr_score(pr_score0, conf.num_nodes)

  var num_pieces = ispace(int1d, conf.num_workers)
  -- compute node and edge partition
  var edge_range = region(num_pieces, regentlib.rect1d)   
  var edge_range_part = partition(equal, edge_range, num_pieces)
  var total_num_edges = init_partition(edge_range, all_edges,
                                       conf.num_edges/ conf.num_workers+1,
                                       conf.num_workers) 
  regentlib.assert(total_num_edges == conf.num_edges, "Edge numbers do not match")

  var part_edges = image(all_edges, edge_range_part, edge_range)
  var part_nodes = image(all_nodes, part_edges, all_edges.dst)
  -- image may not output a disjoint partition
  var part_score0_aliased = image(pr_score0, part_edges, all_edges.dst)
  var cs0 = part_score0_aliased.colors
  -- dynamically cast part_score0_aliased to a disjoint partition part_score0
  var part_score0 = dynamic_cast(partition(disjoint, pr_score0, cs0), part_score0_aliased)
  var part_score1_aliased = image(pr_score1, part_edges, all_edges.dst)
  var cs1 = part_score1_aliased.colors
  var part_score1 = dynamic_cast(partition(disjoint, pr_score1, cs1), part_score1_aliased)

  c.printf("Start PageRank computation...\n")
  var ts_start : int64
  for iter = 0, conf.num_iterations+2 do 
    -- use the first two iterations to warm up the execution
    if iter == 2 then
      __fence(__execution, __block)
      ts_start = c.legion_get_current_time_in_micros()
    end
    if iter % 2 == 0 then
      __demand(__parallel)
      for p in num_pieces do
        pagerank(part_nodes[p], part_edges[p],
                 pr_score0, part_score1[p], 0.9f, conf.num_nodes)
      end
    else
      __demand(__parallel)
      for p in num_pieces do
        pagerank(part_nodes[p], part_edges[p],
                 pr_score1, part_score0[p], 0.9f, conf.num_nodes)
      end
    end
  end
  -- Force all previous tasks to complete before stop the timer
  __fence(__execution, __block)
  var ts_end = c.legion_get_current_time_in_micros()
  c.printf("Iterations = %d, elapsed time = %lldus\n", conf.num_iterations, ts_end - ts_start)
end

regentlib.start(main, bishoplib.make_entry())

