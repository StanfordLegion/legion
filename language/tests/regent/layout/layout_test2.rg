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

-- Compile and link layout_test.cc
local launcher = require("std/launcher")
local clayout_test = launcher.build_library("layout_test")

fspace fs
{
  x : double;
  y : double;
  z : double;
  w : double;
}

task foo(is : ispace(int1d),
         r  : region(is, fs),
         s  : region(is, fs))
where
  reads writes(r, s)
do
  for p in is do
    r[p].x = p * 11
    r[p].y = p * 11
    r[p].z = p * 11
    r[p].w = p * 11
    s[p].x = p * 13
    s[p].y = p * 13
    s[p].z = p * 13
    s[p].w = p * 13
  end
end

task bar(is : ispace(int1d),
         r  : region(is, fs),
         s  : region(is, fs))
where
  reads writes(r, s)
do
  for p in is do
    r[p].x = p * 11
    r[p].y = p * 11
    r[p].z = p * 11
    r[p].w = p * 11
    s[p].x = p * 13
    s[p].y = p * 13
    s[p].z = p * 13
    s[p].w = p * 13
  end
end

task check(is : ispace(int1d),
           r  : region(is, fs),
           s  : region(is, fs),
           t  : region(is, fs),
           w  : region(is, fs))
where
  reads(r, s, t, w)
do
  for p in is do
    regentlib.assert(r[p].x == t[p].x, "test failed")
    regentlib.assert(r[p].y == t[p].y, "test failed")
    regentlib.assert(r[p].z == t[p].z, "test failed")
    regentlib.assert(r[p].w == t[p].w, "test failed")

    regentlib.assert(s[p].x == w[p].x, "test failed")
    regentlib.assert(s[p].y == w[p].y, "test failed")
    regentlib.assert(s[p].z == w[p].z, "test failed")
    regentlib.assert(s[p].w == w[p].w, "test failed")
  end
end

task toplevel()
  var is = ispace(int1d, 13)
  var r  = region(is, fs)
  var s  = region(is, fs)
  var t  = region(is, fs)
  var w  = region(is, fs)
  foo(is, r, s)
  bar(is, t, w)
  check(is, r, s, t, w)
end

local foo_hybrid1 = foo:make_variant("hybrid1")
foo_hybrid1:add_layout_constraint(
  regentlib.layout.ordering_constraint(terralib.newlist {
    regentlib.layout.field_constraint(
      "r",
      terralib.newlist {
        regentlib.field_path("x"),
        regentlib.field_path("z"),
      }
    ),
    regentlib.layout.dimx,
  })
)
foo_hybrid1:add_layout_constraint(
  regentlib.layout.ordering_constraint(terralib.newlist {
    regentlib.layout.dimx,
    regentlib.layout.field_constraint(
      "r",
      terralib.newlist {
        regentlib.field_path("y"),
        regentlib.field_path("w"),
      }
    ),
  })
)
foo_hybrid1:add_layout_constraint(
  regentlib.layout.ordering_constraint(terralib.newlist {
    regentlib.layout.field_constraint(
      "s",
      terralib.newlist {
        regentlib.field_path("x"),
        regentlib.field_path("y"),
        regentlib.field_path("z"),
      }
    ),
    regentlib.layout.dimx,
  })
)
foo_hybrid1:add_layout_constraint(
  regentlib.layout.ordering_constraint(terralib.newlist {
    regentlib.layout.dimx,
    regentlib.layout.field_constraint(
      "s",
      terralib.newlist {
        regentlib.field_path("w"),
      }
    ),
  })
)
regentlib.register_variant(foo_hybrid1)

launcher.launch(toplevel, "layout_test2", clayout_test.register_mappers, {"-llayout_test"})
