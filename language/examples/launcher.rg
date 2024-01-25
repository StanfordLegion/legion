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

-- This file is not meant to be run directly.

-- runs-with:
-- []

local launcher = {}

local root_dir = arg[0]:match(".*/") or "./"

function launcher.build_library(library_name)
  local include_path = ""
  local include_dirs = terralib.newlist()
  include_dirs:insert("-I")
  include_dirs:insert(root_dir)
  for path in string.gmatch(os.getenv("INCLUDE_PATH"), "[^;]+") do
    include_path = include_path .. " -I " .. path
    include_dirs:insert("-I")
    include_dirs:insert(path)
  end

  local library_cc = root_dir .. library_name .. ".cc"
  local library_so
  if os.getenv('OBJNAME') then
    local out_dir = os.getenv('OBJNAME'):match('.*/') or './'
    library_so = out_dir .. "lib" .. library_name .. ".so"
  elseif os.getenv('SAVEOBJ') == '1' then
    library_so = root_dir .. "lib" .. library_name .. ".so"
  else
    -- Make sure we don't collide if we're running this concurrently.
    library_so = os.tmpname() .. ".so"
  end
  local cxx = os.getenv('CXX') or 'c++'

  local cxx_flags = os.getenv('CXXFLAGS') or ''
  cxx_flags = cxx_flags .. " -O2 -Wall -Werror"
  local ffi = require("ffi")
  if ffi.os == "OSX" then
    cxx_flags =
      (cxx_flags ..
         " -dynamiclib -single_module -undefined dynamic_lookup -fPIC")
  else
    cxx_flags = cxx_flags .. " -shared -fPIC"
  end

  local cmd = (cxx .. " " .. cxx_flags .. " " .. include_path .. " " ..
                 library_cc .. " -o " .. library_so)
  if os.execute(cmd) ~= 0 then
    print("Error: failed to compile " .. library_cc)
    assert(false)
  end
  regentlib.linklibrary(library_so)
  return terralib.includec(library_name .. ".h", include_dirs)
end

function launcher.launch(main, default_exe_name, extra_setup_thunk, additional_link_flags)
  if os.getenv('SAVEOBJ') == '1' then
    local root_dir = arg[0]:match(".*/") or "./"
    local out_dir = (os.getenv('OBJNAME') and os.getenv('OBJNAME'):match('.*/')) or root_dir
    local link_flags = terralib.newlist({"-L" .. out_dir})
    link_flags:insertall(additional_link_flags)

    if os.getenv('STANDALONE') == '1' then
      os.execute('cp ' .. os.getenv('LG_RT_DIR') .. '/../bindings/regent/' ..
          regentlib.binding_library .. ' ' .. out_dir)
    end

    local exe = os.getenv('OBJNAME') or default_exe_name
    regentlib.saveobj(main, exe, "executable", extra_setup_thunk, link_flags)
  else
    regentlib.start(main, extra_setup_thunk)
  end
end

return launcher
