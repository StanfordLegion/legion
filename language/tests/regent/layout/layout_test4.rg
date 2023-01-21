-- Copyright 2023 Stanford University, Los Alamos National Laboratory
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

fspace vec
{
  a : double;
  b : double;
}

fspace fs
{
  c : vec;
  d : vec;
}

task foo(is : ispace(int2d),
         r  : region(is, fs),
         s  : region(is, fs))
where
  reads writes(r, s)
do
  for p in is do
    r[p].c.a = p.x * 3 + p.y * 11
    r[p].c.b = p.x * 3 + p.y * 11
    r[p].d.a = p.x * 3 + p.y * 11
    r[p].d.b = p.x * 3 + p.y * 11
    s[p].c.a = p.x * 7 + p.y * 13
    s[p].c.b = p.x * 7 + p.y * 13
    s[p].d.a = p.x * 7 + p.y * 13
    s[p].d.b = p.x * 7 + p.y * 13
  end
end

task bar(is : ispace(int2d),
         r  : region(is, fs),
         s  : region(is, fs))
where
  reads writes(r, s)
do
  for p in is do
    r[p].c.a = p.x * 3 + p.y * 11
    r[p].c.b = p.x * 3 + p.y * 11
    r[p].d.a = p.x * 3 + p.y * 11
    r[p].d.b = p.x * 3 + p.y * 11
    s[p].c.a = p.x * 7 + p.y * 13
    s[p].c.b = p.x * 7 + p.y * 13
    s[p].d.a = p.x * 7 + p.y * 13
    s[p].d.b = p.x * 7 + p.y * 13
  end
end

task check(is : ispace(int2d),
           r  : region(is, fs),
           s  : region(is, fs),
           t  : region(is, fs),
           w  : region(is, fs))
where
  reads(r, s, t, w)
do
  for p in is do
    regentlib.assert(r[p].c.a == t[p].c.a, "test failed")
    regentlib.assert(r[p].c.b == t[p].c.b, "test failed")
    regentlib.assert(r[p].d.a == t[p].d.a, "test failed")
    regentlib.assert(r[p].d.b == t[p].d.b, "test failed")

    regentlib.assert(s[p].c.a == w[p].c.a, "test failed")
    regentlib.assert(s[p].c.b == w[p].c.b, "test failed")
    regentlib.assert(s[p].d.a == w[p].d.a, "test failed")
    regentlib.assert(s[p].d.b == w[p].d.b, "test failed")
  end
end

task toplevel()
  var is = ispace(int2d, {5, 10})
  var r  = region(is, fs)
  var s  = region(is, fs)
  var t  = region(is, fs)
  var w  = region(is, fs)
  foo(is, r, s)
  bar(is, t, w)
  check(is, r, s, t, w)
end

local foo_primary_variant = foo:get_primary_variant()
foo_primary_variant:add_layout_constraint(
  regentlib.layout.ordering_constraint(terralib.newlist {
    regentlib.layout.field_constraint(
      "r",
      terralib.newlist {
        regentlib.field_path("d", "a"),
        regentlib.field_path("c", "b"),
      }
    ),
    regentlib.layout.dimy,
    regentlib.layout.dimx,
  })
)
foo_primary_variant:add_layout_constraint(
  regentlib.layout.ordering_constraint(terralib.newlist {
    regentlib.layout.dimy,
    regentlib.layout.dimx,
    regentlib.layout.field_constraint(
      "r",
      terralib.newlist {
        regentlib.field_path("d", "b"),
        regentlib.field_path("c", "a"),
      }
    ),
  })
)
foo_primary_variant:add_layout_constraint(
  regentlib.layout.ordering_constraint(terralib.newlist {
    regentlib.layout.field_constraint(
      "s",
      terralib.newlist {
        regentlib.field_path("c"),
        regentlib.field_path("d", "b"),
      }
    ),
    regentlib.layout.dimx,
    regentlib.layout.dimy,
  })
)
foo_primary_variant:add_layout_constraint(
  regentlib.layout.ordering_constraint(terralib.newlist {
    regentlib.layout.dimx,
    regentlib.layout.dimy,
    regentlib.layout.field_constraint(
      "s",
      terralib.newlist {
        regentlib.field_path("d", "a"),
      }
    ),
  })
)

regentlib.start(toplevel)
