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

local c = regentlib.c

struct GraphCfg {
  nodes : int,
  edges : int,
  datafile : int8[256],
  num_sources : int,
  sources : &int,
  expecteds : &(int8[256]),
}

terra GraphCfg:read_config_file(dirname: &int8)
  var fn : int8[80]
  c.sprintf(fn, "%s/graph.txt", dirname)
  var fh = c.fopen(fn, "r")
  c.fscanf(fh, "nodes %d\n", &self.nodes)
  c.fscanf(fh, "edges %d\n", &self.edges)
  -- prepend directory
  var len = c.sprintf(self.datafile, "%s/", dirname)
  c.fscanf(fh, "data %s\n", self.datafile+len)

  var i = 0
  self.sources = [&int](0)
  self.expecteds = [&(int8[256])](0)
  while c.feof(fh) == 0 do
    var id : int
    var s : int[256]
    var n = c.fscanf(fh, "source %d %s\n", &id, s)
    if n ~= 2 then
      break
    end
    self.sources = [&int](c.realloc(self.sources, (i+1) * sizeof(int)))
    self.expecteds = [&(int8[256])](c.realloc(self.expecteds, (i+1) * 258 * sizeof(int8)));
    self.sources[i] = id
    c.sprintf(self.expecteds[i], "%s/%s", dirname, s)
    i = i + 1
  end
  self.num_sources = i
    
  c.fclose(fh)
  return self
end
--GraphCfg.methods['read_config_file']:compile()

terra GraphCfg:show()
  c.printf("graph metadata: nodes=%d, edges=%d, sources=%d\n",
	   self.nodes, self.edges, self.num_sources)
end
--GraphCfg.methods['show']:compile()

return GraphCfg
