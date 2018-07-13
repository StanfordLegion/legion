-- Copyright 2018 Stanford University, Los Alamos National Laboratory
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

-- Compile and link layout_test1.cc
local clayout_test1
do
  local root_dir = arg[0]:match(".*/") or "./"
  local runtime_dir = os.getenv('LG_RT_DIR') .. "/"
  local layout_test1_cc = root_dir .. "layout_test1.cc"
  local layout_test1_so
  if os.getenv('OBJNAME') then
    local out_dir = os.getenv('OBJNAME'):match('.*/') or './'
    layout_test1_so = out_dir .. "liblayout_test1.so"
  elseif os.getenv('SAVEOBJ') == '1' then
    layout_test1_so = root_dir .. "liblayout_test1.so"
  else
    layout_test1_so = os.tmpname() .. ".so" -- root_dir .. "layout_test1.so"
  end
  local cxx = os.getenv('CXX') or 'c++'

  local cxx_flags = os.getenv('CC_FLAGS') or ''
  --cxx_flags = cxx_flags .. " -O2 -Wall -Werror"
  cxx_flags = cxx_flags .. " -g -O0"
  if os.execute('test "$(uname)" = Darwin') == 0 then
    cxx_flags =
      (cxx_flags ..
         " -dynamiclib -single_module -undefined dynamic_lookup -fPIC")
  else
    cxx_flags = cxx_flags .. " -shared -fPIC"
  end

  local cmd = (cxx .. " " .. cxx_flags .. " -I " .. runtime_dir .. " " ..
                 layout_test1_cc .. " -o " .. layout_test1_so)
  if os.execute(cmd) ~= 0 then
    print("Error: failed to compile " .. layout_test1_cc)
    assert(false)
  end
  terralib.linklibrary(layout_test1_so)
  clayout_test1 = terralib.includec("layout_test1.h", {"-I", root_dir, "-I", runtime_dir})
end

fspace fs
{
  x : double;
  y : double;
  z : double;
  w : double;
}

task foo(r : region(ispace(int2d), fs),
         s : region(ispace(int2d), fs))
where
  reads writes(r)
do
end

task toplevel()
  var r = region(ispace(int2d, {5, 10}), fs)
  var s = region(ispace(int2d, {5, 10}), fs)
  foo(r, s)
end

local foo_hybrid1 = foo:make_variant("hybrid1")
foo_hybrid1:add_layout_constraint(
  regentlib.layout.ordering_constraint(terralib.newlist {
    regentlib.layout.field_constraint(
      "r",
      terralib.newlist {
        regentlib.layout.field_path("x"),
        regentlib.layout.field_path("z"),
      }
    ),
    regentlib.layout.dimx,
    regentlib.layout.dimy,
    regentlib.layout.dimz,
  })
)
foo_hybrid1:add_layout_constraint(
  regentlib.layout.ordering_constraint(terralib.newlist {
    regentlib.layout.dimx,
    regentlib.layout.dimy,
    regentlib.layout.dimz,
    regentlib.layout.field_constraint(
      "r",
      terralib.newlist {
        regentlib.layout.field_path("y"),
        regentlib.layout.field_path("w"),
      }
    ),
  })
)
foo_hybrid1:add_layout_constraint(
  regentlib.layout.ordering_constraint(terralib.newlist {
    regentlib.layout.dimx,
    regentlib.layout.dimy,
    regentlib.layout.dimz,
    regentlib.layout.field_constraint(
      "s",
      terralib.newlist {
        regentlib.layout.field_path("x"),
        regentlib.layout.field_path("y"),
        regentlib.layout.field_path("z"),
      }
    ),
  })
)
foo_hybrid1:add_layout_constraint(
  regentlib.layout.ordering_constraint(terralib.newlist {
    regentlib.layout.dimx,
    regentlib.layout.dimy,
    regentlib.layout.dimz,
    regentlib.layout.field_constraint(
      "s",
      terralib.newlist {
        regentlib.layout.field_path("w"),
      }
    ),
  })
)
regentlib.register_variant(foo_hybrid1)

regentlib.start(toplevel, clayout_test1.register_mappers)
