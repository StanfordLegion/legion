-- Copyright 2023 Stanford University
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
-- [
--  ["-ll:cpu", "3", "-fbounds-checks", "1",
--   "-fparallelize-dop", "9", "-fflow-spmd", "1", "-fflow-spmd-shardsize", "3"],
--  ["-ll:cpu", "4", "-fflow-spmd", "1"],
--  ["-ll:cpu", "4", "-fflow-spmd", "1", "-fflow-spmd-shardsize", "2"]
-- ]

import "regent"

fspace fs
{
  f : int32,
  g : int32,
  h : int32,
}

__demand(__parallel)
task centered(r : region(ispace(int3d), fs))
where reads writes(r.f), reads(r.g)
do
  for e in r do
    r[e].f = r[e].g
  end
end

__demand(__parallel)
task stencil1(r : region(ispace(int3d), fs))
where reads(r.f), reads writes(r.g)
do
  for e in r do
    r[e].g = r[(e + {-1,  0,  0}) % r.bounds].f
  end
end

__demand(__parallel)
task stencil2(r : region(ispace(int3d), fs))
where reads(r.f), reads writes(r.h)
do
  for e in r do
    r[e].h = r[(e + { 1,  0,  0}) % r.bounds].f
  end
end

task test(size : int)
  var is = ispace(int3d, {size, size, size})
  var primary_region = region(is, fs)

  fill(primary_region.f, 1)
  fill(primary_region.g, 2)
  fill(primary_region.h, 3)

  var steps = 1
  __demand(__spmd)
  for i = 0, steps do
    stencil1(primary_region)
    centered(primary_region)
    stencil2(primary_region)
  end
end

task toplevel()
  test(10)
end

regentlib.start(toplevel)
