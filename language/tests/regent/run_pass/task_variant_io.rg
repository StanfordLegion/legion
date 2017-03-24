-- Copyright 2017 Stanford University
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
  local runtime_dir = root_dir .. "../../../../runtime/"
  local legion_dir = runtime_dir .. "legion/"
  local mapper_dir = runtime_dir .. "mappers/"
  local realm_dir = runtime_dir .. "realm/"
  local mapper_cc = root_dir .. "task_variant_io.cc"
  if os.getenv('SAVEOBJ') == '1' then
    mapper_so = root_dir .. "libtask_variant_io.so"
  else
    mapper_so = os.tmpname() .. ".so" -- root_dir .. "task_variant_io.so"
  end
  local cxx = os.getenv('CXX') or 'c++'

  local cxx_flags = "-O2 -Wall -Werror"
  if os.execute('test "$(uname)" = Darwin') == 0 then
    cxx_flags =
      (cxx_flags ..
         " -dynamiclib -single_module -undefined dynamic_lookup -fPIC")
  else
    cxx_flags = cxx_flags .. " -shared -fPIC"
  end

  local cmd = (cxx .. " " .. cxx_flags .. " -I " .. runtime_dir .. " " ..
                 " -I " .. mapper_dir .. " " .. " -I " .. legion_dir .. " " ..
                 " -I " .. realm_dir .. " " .. mapper_cc .. " -o " .. mapper_so)
  if os.execute(cmd) ~= 0 then
    print(cmd)
    print("Error: failed to compile " .. mapper_cc)
    assert(false)
  end
  terralib.linklibrary(mapper_so)
  cmapper = terralib.includec("task_variant_io.h", {"-I", root_dir, "-I", runtime_dir,
                                                   "-I", mapper_dir, "-I", legion_dir,
                                                   "-I", realm_dir})
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

task main()
  f()
end
regentlib.start(main, cmapper.register_mappers)
