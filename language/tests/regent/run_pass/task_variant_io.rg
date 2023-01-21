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
-- [["-ll:io", "1"]]

import "regent"

do
  local root_dir = arg[0]:match(".*/") or "./"

  local include_path = ""
  local include_dirs = terralib.newlist()
  include_dirs:insert("-I")
  include_dirs:insert(root_dir)
  for path in string.gmatch(os.getenv("INCLUDE_PATH"), "[^;]+") do
    include_path = include_path .. " -I " .. path
    include_dirs:insert("-I")
    include_dirs:insert(path)
  end

  local mapper_cc = root_dir .. "task_variant_io.cc"
  if os.getenv('SAVEOBJ') == '1' then
    mapper_so = root_dir .. "libtask_variant_io.so"
  else
    mapper_so = os.tmpname() .. ".so" -- root_dir .. "task_variant_io.so"
  end
  local cxx = os.getenv('CXX') or 'c++'

  local cxx_flags = os.getenv('CXXFLAGS') or ''
  cxx_flags = cxx_flags .. " -O2 -Wall -Werror"
  if os.execute('test "$(uname)" = Darwin') == 0 then
    cxx_flags =
      (cxx_flags ..
         " -dynamiclib -single_module -undefined dynamic_lookup -fPIC")
  else
    cxx_flags = cxx_flags .. " -shared -fPIC"
  end

  local cmd = (cxx .. " " .. cxx_flags .. " " .. include_path .. " " ..
                 mapper_cc .. " -o " .. mapper_so)
  if os.execute(cmd) ~= 0 then
    print(cmd)
    print("Error: failed to compile " .. mapper_cc)
    assert(false)
  end
  regentlib.linklibrary(mapper_so)
  cmapper = terralib.includec("task_variant_io.h", include_dirs)
end

local c = regentlib.c

-- Right now there's no way to force this to run on an IO processor
-- but with the default kind ranking it should get assigned automatically.
task f()
  var proc =
    c.legion_runtime_get_executing_processor(__runtime(), __context())
  c.printf("executing on processor %llx\n", proc.id)
  regentlib.assert(c.legion_processor_kind(proc) == c.IO_PROC, "test failed")
end
-- This is here to test that there isn't a duplicate registration in
-- the case where the user sets an explicit variant ID.
f:get_primary_variant():set_variant_id(123)

task main()
  f()
end
regentlib.start(main, cmapper.register_mappers)
