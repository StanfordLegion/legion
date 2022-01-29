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

import "regent"

local c = regentlib.c
local cstring = terralib.includec("string.h")
local std = terralib.includec("stdlib.h")

struct CktConfig {
  num_loops : int,
  num_pieces : int,
  nodes_per_piece : int,
  wires_per_piece : int,
  steps : int,
  pct_wire_in_piece : int,
  random_seed : int,
  dump_graph : bool,
}

terra CktConfig:initialize_from_command()
  self.num_loops = 5
  self.num_pieces = 4
  self.nodes_per_piece = 4
  self.wires_per_piece = 8
  self.pct_wire_in_piece = 80
  self.random_seed = 12345
  self.steps = 10000
  self.dump_graph = false

  var args = c.legion_runtime_get_input_args()
  var i = 0
  while i < args.argc do
    if cstring.strcmp(args.argv[i], "-l") == 0 then
      i = i + 1
      self.num_loops = std.atoi(args.argv[i])
    elseif cstring.strcmp(args.argv[i], "-i") == 0 then
      i = i + 1
      self.steps = std.atoi(args.argv[i])
    elseif cstring.strcmp(args.argv[i], "-p") == 0 then
      i = i + 1
      self.num_pieces = std.atoi(args.argv[i])
    elseif cstring.strcmp(args.argv[i], "-npp") == 0 then
      i = i + 1
      self.nodes_per_piece = std.atoi(args.argv[i])
    elseif cstring.strcmp(args.argv[i], "-wpp") == 0 then
      i = i + 1
      self.wires_per_piece = std.atoi(args.argv[i])
    elseif cstring.strcmp(args.argv[i], "-pct") == 0 then
      i = i + 1
      self.pct_wire_in_piece = std.atoi(args.argv[i])
    elseif cstring.strcmp(args.argv[i], "-s") == 0 then
      i = i + 1
      self.random_seed = std.atoi(args.argv[i])
    elseif cstring.strcmp(args.argv[i], "-dumpgraph") == 0 then
      i = i + 1
      self.dump_graph = true
    end
    i = i + 1
  end
end

terra CktConfig:show()
  c.printf(["circuit settings: loops=%d pieces=%d " ..
            "nodes/piece=%d wires/piece=%d " ..
            "pct_in_piece=%d seed=%d\n"],
    self.num_loops, self.num_pieces,
    self.nodes_per_piece, self.wires_per_piece,
    self.pct_wire_in_piece, self.random_seed)
end

return CktConfig
