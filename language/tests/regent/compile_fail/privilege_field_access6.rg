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

-- fails-with:
-- privilege_field_access6.rg:43: invalid privilege reads($rp.pu.x) for dereference of ptr(point, $rp)
--     p.px.x = p.px0.x + dth*(p.pu.x + p.pu0.x)
--                             ^

import "regent"

struct vec2 {
  x : double,
  y : double,
}

struct point {
  px : vec2,
  px0 : vec2,
  pu : vec2,
  pu0 : vec2,
}

task adv_pos_full(rp : region(point),
                  dt : double)
where
  reads(rp.{px0, pu0}),
  writes(rp.{px, pu})
do
  var dth = 0.5 * dt
  for p in rp do
    p.pu.x = p.pu0.x
    p.px.x = p.px0.x + dth*(p.pu.x + p.pu0.x)
  end
end
adv_pos_full:compile()
