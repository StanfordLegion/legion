-- Copyright 2024 Stanford University, NVIDIA Corporation
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

-- From issue #529

local c = regentlib.c

fspace Cluster {
  cluster: int3d,
  bounds: rect2d,
  sep: int2d,
}

task partition_by_image_range(mat:region(ispace(int2d), double),
                              clusters:region(ispace(int3d), Cluster),
                              cluster_part:partition(disjoint, clusters, ispace(int3d)))
where
  reads(clusters.bounds)
do
  var fid = __fields(clusters.bounds)[0]

  var ip = c.legion_index_partition_create_by_image_range(__runtime(),
                                                          __context(),
                                                          __raw(mat.ispace),
                                                          __raw(cluster_part),
                                                          __raw(clusters),
                                                          fid,
                                                          __raw(clusters.ispace),
                                                          c.DISJOINT_KIND,
                                                          c.AUTO_GENERATE_ID,
                                                          0,
                                                          0,
                                                          c.legion_untyped_buffer_t {nil, 0})

  var raw_part = c.legion_logical_partition_create(__runtime(), __raw(mat), ip)

  return __import_partition(disjoint, mat, clusters.ispace, raw_part)
end

task main()
  var mat = region(ispace(int2d, {400, 400}), double)

  var blocks = region(ispace(int3d, {31, 31, 64}, {1, 1, 0}), int1d)

  blocks[{25, 25, 0}] = 1
  blocks[{25, 25, 1}] = 1

  blocks[{31, 18, 0}] = 1
  blocks[{31, 18, 1}] = 1

  var allocated_blocks_part = partition(blocks, ispace(int1d, 2))
  var allocated_blocks_ispace = allocated_blocks_part[1].ispace

  var clusters = region(allocated_blocks_ispace, Cluster)
  fill(clusters.bounds, rect2d{lo = int2d{0, 0}, hi = int2d{-1, -1}})

  for color in clusters.ispace do
    clusters[color].sep = int2d{color.x, color.y}
    clusters[color].cluster = color
  end

  var cpart = partition(clusters.cluster, allocated_blocks_ispace)

  clusters[{25, 25, 0}].bounds = rect2d{ lo = int2d{0, 0}, hi = int2d{0, 0}}
  clusters[{25, 25, 1}].bounds = rect2d{ lo = int2d{0, 1}, hi = int2d{0, 3}}

  clusters[{31, 18, 0}].bounds = rect2d{ lo = int2d{380, 293}, hi = int2d{380, 296}}
  clusters[{31, 18, 1}].bounds = rect2d{ lo = int2d{381, 293}, hi = int2d{381, 296}}

  var mat_part = partition_by_image_range(mat, clusters, cpart)

  for i in mat_part.colors do
    var r = mat_part[i]
    if r.volume ~= 0 then
      var lo = r.bounds.lo
      var hi = r.bounds.hi
      c.printf("Color: %d %d %d Lo: %d %d Hi: %d %d Vol: %d\n", i.x, i.y, i.z, lo.x, lo.y, hi.x, hi.y, r.volume)
    end
  end
end

regentlib.start(main)
