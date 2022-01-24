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

-- runs-with:
-- []

import "regent"
import "bishop"

mapper

$GPUs = processors[isa=cuda]
$CPUs = processors[isa=x86]
$HAS_ZCMEM = memories[kind=zcmem].size > 0
$HAS_GPUS = $GPUs.size > 0

task#init_pr_score[index=$i],
task#pagerank[index=$i] {
  target : $GPUs[$i % $GPUs.size];
}

task#init_graph[index=$i],
task#init_partition[index=$i] {
  target : $CPUs[$i % $CPUs.size];
}

task#pagerank[target=$proc] region#nodes,
task#pagerank[target=$proc] region#edges,
task#pagerank[target=$proc] region#pr_ws{
  target : $proc.memories[kind=fbmem];
}

task#init_pr_score[target=$proc] region#pr,
task#pagerank[target=$proc] region#pr_old,
task#pagerank[target=$proc] region#pr_new {
  target : $proc.memories[kind=zcmem];
}

task#init_graph[target=$proc] region#nodes,
task#init_graph[target=$proc] region#edges,
task#init_partition[target=$proc] region#node_range,
task#init_partition[target=$proc] region#edge_range,
task#init_partition[target=$proc] region#nodes {
  target : $proc.memories[kind=zcmem];
}

end

local c = regentlib.c
local std = terralib.includec("stdlib.h")
local cstring = terralib.includec("string.h")
local V_ID = int32
local E_ID = int64
rawset(_G, "rand", std.rand)

struct Config {
  num_nodes : V_ID,
  num_edges : E_ID,
  num_iterations : int64,
  num_workers : int64,
  graph : int8[256],
}

struct NodeStruct {
  index : E_ID,
  degree : V_ID,
}

struct EdgeStruct {
  src : V_ID,
  dst : V_ID,
}

local clegion_interop
do
  assert(os.getenv('LG_RT_DIR') ~= nil, "$LG_RT_DIR should be set!")
  local root_dir = arg[0]:match(".*/") or "./"
  local runtime_dir = os.getenv('LG_RT_DIR') .. "/"
  local cubpath_dir = os.getenv("CUB_INCLUDE_DIR") or "/share/software/user/open/cub/1.7.3/"
  local legion_dir = runtime_dir .. "legion/"
  local mapper_dir = runtime_dir .. "mappers/"
  local realm_dir = runtime_dir .. "realm/"
  local legion_interop_cc = root_dir .. "legion_interop.cu"
  local legion_interop_so
  --if os.getenv('SAVEOBJ') == '1' then
  if true then
    legion_interop_so = root_dir .. "liblegion_interop.so"
  else
    legion_interop_so = os.tmpname() .. ".so" -- root_dir .. "mapper.so"
  end
  local cxx = os.getenv('NVCC') or 'nvcc'

  local cxx_flags = os.getenv('CXXFLAGS') or ''
  -- cxx_flags = cxx_flags
  if os.execute('test "$(uname)" = Darwin') == 0 then
    cxx_flags =
      (cxx_flags ..
         " -dynamiclib -single_module -undefined dynamic_lookup -std=c++11 -Xcompiler -fPIC")
  else
    cxx_flags = cxx_flags .. " -shared -arch=compute_60 -code=sm_60 -std=c++11 -Xcompiler -fPIC"
  end

  local cmd = (cxx .. " " .. cxx_flags .. " -I " .. runtime_dir .. " " ..
                 " -I " .. mapper_dir .. " " .. " -I " .. legion_dir .. " " .. " -I " .. cubpath_dir .. " " ..
                 " -I " .. realm_dir .. " " .. legion_interop_cc .. " -o " .. legion_interop_so)
  if os.execute(cmd) ~= 0 then
    print("Error: failed to compile " .. legion_interop_cc)
    assert(false)
  end
  regentlib.linklibrary(legion_interop_so)
  clegion_interop =
    terralib.includec("legion_interop.h", {"-I", root_dir, "-I", runtime_dir,
                                           "-I", mapper_dir, "-I", legion_dir,
                                           "-I", realm_dir})
end

terra parse_input_args(conf : Config)
  var args = c.legion_runtime_get_input_args()
  var input_file : rawstring = nil
  for i = 0, args.argc do
    if cstring.strcmp(args.argv[i], "-ni") == 0 then
      i = i + 1
      conf.num_iterations = std.atoll(args.argv[i])
    elseif cstring.strcmp(args.argv[i], "-nw") == 0 then
      i = i + 1
      conf.num_workers = std.atoll(args.argv[i])
    elseif cstring.strcmp(args.argv[i], "-graph") == 0 then
      i = i + 1
      input_file = rawstring(args.argv[i])
    end
  end
  regentlib.assert_error(input_file ~= nil, "error: the flag '-graph' is required but was not provided")
  var file = c.fopen(input_file, "r")
  regentlib.assert(file ~= nil, "failed to open input file")
  c.fscanf(file, "%i", &conf.num_nodes)
  c.fscanf(file, "%i", &conf.num_edges)
  c.fscanf(file, "%255s", conf.graph)
  c.fclose(file)
  return conf
end

task init_graph(nodes : region(ispace(int1d), NodeStruct),
                edges : region(ispace(int1d), EdgeStruct),
                num_nodes : V_ID,
                num_edges : E_ID,
                graph : regentlib.string)
where
  reads(nodes, edges), writes(nodes, edges)
do
  var indices : &E_ID = [&E_ID](c.malloc(num_nodes * 8))
  var degrees : &V_ID = [&V_ID](c.malloc(num_nodes * 4))
  var srcs : &V_ID = [&V_ID](c.malloc(num_edges * 4))
  var file = c.fopen([rawstring](graph), "rb")
  regentlib.assert(not isnull(file), "failed to open graph file")
  c.printf("graph = %s\n", [rawstring](graph))
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
    edges[e].dst = dst
  end
  c.fclose(file)
  c.free(indices)
  c.free(degrees)
  c.free(srcs)
  
  return 1
end

task init_partition(node_range : region(ispace(int1d), regentlib.rect1d),
                    edge_range : region(ispace(int1d), regentlib.rect1d),
                    nodes : region(ispace(int1d), NodeStruct),
                    avg_num_edges : E_ID,
                    num_parts : int)
where
  writes(node_range, edge_range), reads(nodes)
do
  var range_is = node_range.ispace
  var node_is = nodes.ispace
  var total_num_edges : E_ID = 0
  var start_idx : E_ID = 0
  var start_node : V_ID = 0
  var p : int = 0
  for n in node_is do
    if ((nodes[n].index - start_idx > avg_num_edges) or (n == node_is.bounds.hi)) then
      node_range[p] = {start_node, n}
      edge_range[p] = {start_idx, nodes[n].index - 1}
      start_idx = nodes[n].index
      start_node = n + 1
      p = p + 1
    end
  end
  regentlib.assert(p == range_is.volume, "Number of partitions don't match")
  return nodes[node_is.bounds.hi].index
end

extern task pagerank(nodes : region(ispace(int1d), NodeStruct),
                     edges : region(ispace(int1d), EdgeStruct),
                     pr_old : region(ispace(int1d), float),
                     pr_new : region(ispace(int1d), float),
                     pr_ws : region(ispace(int1d), float))
where
  reads(nodes, edges, pr_old), writes(pr_new, pr_ws)
end
pagerank:set_task_id(clegion_interop.TID_F)
pagerank:set_calling_convention(regentlib.convention.manual())

extern task init_pr_score(pr : region(ispace(int1d), float))
where
  writes(pr)
end
init_pr_score:set_task_id(clegion_interop.TID_F2)
init_pr_score:set_calling_convention(regentlib.convention.manual())

terra check(x : int)
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
  var is_workspace = ispace(int1d, conf.num_nodes * conf.num_workers)

  var all_nodes = region(is_nodes, NodeStruct)
  var all_edges = region(is_edges, EdgeStruct)

  var pr_score0 = region(is_nodes, float)
  var pr_score1 = region(is_nodes, float)
  var pr_workspace = region(is_workspace, float)

  c.printf("Load input graph...\n")
  init_graph(all_nodes, all_edges, conf.num_nodes, conf.num_edges, [rawstring](conf.graph))
  init_pr_score(pr_score0)

  var pieces = ispace(int1d, conf.num_workers)
  var part_workspace = partition(equal, pr_workspace, pieces)
  -- compute node and edge partition
  var node_range = region(pieces, regentlib.rect1d)
  var edge_range = region(pieces, regentlib.rect1d)
  var part_node_range = partition(equal, node_range, pieces)
  var part_edge_range = partition(equal, edge_range, pieces)
  var total_num_edges = init_partition(node_range, edge_range, all_nodes,
                                       conf.num_edges/ conf.num_workers+1,
                                       conf.num_workers) 
  regentlib.assert(total_num_edges == conf.num_edges, "Edge number not match")
  __fence(__execution, __block)
  var part_nodes = image(all_nodes, part_node_range, node_range)
  var part_score0 = image(disjoint, pr_score0, part_node_range, node_range)
  var part_score1 = image(disjoint, pr_score1, part_node_range, node_range)
  var part_edges = image(all_edges, part_edge_range, edge_range)

  __fence(__execution, __block)
  c.printf("Start PageRank computation...\n")
  var ts_start : int64
  for iter = 0, conf.num_iterations+2 do
    -- use the first two iterations to warm up the execution
    if iter == 2 then
      __fence(__execution, __block)
      ts_start = c.legion_get_current_time_in_micros()
    end
    if iter % 2 == 0 then
      __demand(__index_launch)
      for p in pieces do
        pagerank(part_nodes[p], part_edges[p],
                 pr_score0, part_score1[p], part_workspace[p])
      end
    else
      __demand(__index_launch)
      for p in pieces do
        pagerank(part_nodes[p], part_edges[p],
                 pr_score1, part_score0[p], part_workspace[p])
      end
    end
  end
  -- Force all previous tasks to complete before stop the timer
  __fence(__execution, __block)
  var ts_end = c.legion_get_current_time_in_micros()
  c.printf("Iterations = %d, elapsed time = %lldus\n", conf.num_iterations, ts_end - ts_start)
end

terra callback()
  clegion_interop.register_tasks()
  [bishoplib.make_entry()]()
end

regentlib.start(main, callback)
