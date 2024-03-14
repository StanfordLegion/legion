-- Copyright 2024 Stanford University
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
import "bishop"

local c = regentlib.c

-- Simple test to trigger 2D (and 3D) copies.

mapper

$CPUS = processors[isa=x86]
$GPUS = processors[isa=cuda]
$HAS_GPUS = $GPUS.size > 0

task#producer_x {
  target : $CPUS[0];
}

task#consumer_x {
  target : $HAS_GPUS ? $GPUS[0] : $CPUS[0];
}

task[isa=cuda and target=$p] region {
  target : $p.memories[kind=fbmem];
}

task[isa=x86 and target=$p] region {
  target : $p.memories[kind=sysmem];
}

end

fspace fs {
  x : int,
  y : int,
}

task producer_x(r : region(ispace(int3d), fs))
where writes(r.x) do
  return 0
end

task producer_y(r : region(ispace(int3d), fs))
where writes(r.y) do
  return 0
end

__demand(__cuda)
task consumer_x(r : region(ispace(int3d), fs))
where reads(r.x) do
  return 0
end

task consumer_xy(r : region(ispace(int3d), fs))
where reads(r.{x,y}) do
  return 0
end

task make_subregion(r : region(ispace(int3d), fs),
                    lo : int3d, hi : int3d)
  var coloring = c.legion_domain_point_coloring_create()
  var cspace = ispace(int1d, 1)
  var i : int1d = 0
  var rect = rect3d { lo = lo, hi = hi }
  c.legion_domain_point_coloring_color_domain(coloring, i, rect)
  var p = partition(disjoint, r, coloring, cspace)
  c.legion_domain_point_coloring_destroy(coloring)
  return p
end

task dotest(size : int3d,
            p_lo : int3d, p_hi : int3d,
	    c_lo : int3d, c_hi : int3d)
  var is = ispace(int3d, size)
  var r = region(is, fs)
  fill(r.{x, y}, 0)
  var p_p = make_subregion(r, p_lo, p_hi)
  var p_c = make_subregion(r, c_lo, c_hi)
  var t1 = c.legion_get_current_time_in_micros()
  var k1 = producer_x(p_p[0])
  if k1 > 0 then c.printf("FAIL\n") end
  var t2 = c.legion_get_current_time_in_micros()
  var k2 = consumer_x(p_c[0])
  if k2 > 0 then c.printf("FAIL\n") end
  var t3 = c.legion_get_current_time_in_micros()
  return t3-t2
end

task main()
  var n = 1024 * 1024  -- 4MB transferred total
  var nx = 16
  var ny = n / nx

  -- 1d copy by doing a strip in the first dimension
  var t_1d = dotest({ x = n+2, y = 1, z = 1 },
                    { x = 1, y = 0, z = 0 },
		    { x = n, y = 0, z = 0 },
		    { x = 0, y = 0, z = 0 },
		    { x = n+1, y = 0, z = 0 })
  var bw_1d = 4.0 * n / t_1d
  c.printf("1d: ms=%lld bw=%g\n", t_1d, bw_1d)

  -- 2d copy by doing a strip in the second dimension
  var t_2d = dotest({ x = 2, y = n+2, z = 1 },
                    { x = 0, y = 1, z = 0 },
		    { x = 1, y = n, z = 0 },
		    { x = 0, y = 0, z = 0 },
		    { x = 0, y = n+1, z = 0 })
  var bw_2d = 4.0 * n / t_2d
  c.printf("2d: ms=%lld bw=%g\n", t_2d, bw_2d)

  -- 3d needs a rectangle in the second and third dimensions
  var t_3d = dotest({ x = 2, y = nx+2, z = ny+2 },
                    { x = 0, y = 1, z = 1 },
		    { x = 1, y = nx, z = ny },
		    { x = 0, y = 0, z = 0 },
		    { x = 0, y = nx+1, z = ny+1 })
  var bw_3d = 4.0 * n / t_3d
  c.printf("3d: ms=%lld bw=%g\n", t_3d, bw_3d)

  if nx ~= ny then
    -- 3d needs a rectangle in the second and third dimensions
    var t_3d = dotest({ x = 2, y = ny+2, z = nx+2 },
                      { x = 0, y = 1, z = 1 },
		      { x = 1, y = ny, z = nx },
		      { x = 0, y = 0, z = 0 },
		      { x = 0, y = ny+1, z = nx+1 })
    var bw_3d = 4.0 * n / t_3d
    c.printf("3d: ms=%lld bw=%g\n", t_3d, bw_3d)
  end
end
regentlib.start(main, bishoplib.make_entry())
