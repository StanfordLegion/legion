local launcher = {}

local root_dir = arg[0]:match(".*/") or "./"
local runtime_dir = os.getenv("LG_RT_DIR") or (root_dir .. "../../runtime")
local legion_dir = runtime_dir .. "/legion"
local realm_dir = runtime_dir .. "/realm"
local mapper_dir = runtime_dir .. "/mappers"

function launcher.compile_mapper(saveobj, prefix)
  local cc_file = root_dir .. prefix .. ".cc"
  local binary_file

  if saveobj then
    binary_file = root_dir .. "lib" .. prefix .. ".so"
  else
    binary_file = os.tmpname() .. ".so"
  end

  local cxx_flags = "-O2 -std=c++0x -Wall -Werror"
  if os.execute('test "$(uname)" = Darwin') == 0 then
    cxx_flags =
    (cxx_flags ..
    " -dynamiclib -single_module -undefined dynamic_lookup -fPIC")
  else
    cxx_flags = cxx_flags .. " -shared -fPIC"
  end
  local cxx = os.getenv('CXX') or 'c++'

  local cmd = (cxx .. " " .. cxx_flags ..
              " -I " .. runtime_dir ..
              " -I " .. mapper_dir ..
              " -I " .. legion_dir ..
              " -I " .. realm_dir .. " " ..
              cc_file .. " -o " .. binary_file)
  if os.execute(cmd) ~= 0 then
    print("Error: failed to compile " .. cc_file)
    assert(false)
  end
  return binary_file
end

function launcher.launch(toplevel, prefix)
  local saveobj = os.getenv('SAVEOBJ') == '1'
  local mapper_binary = launcher.compile_mapper(saveobj, prefix)
  local mapper_header = terralib.includec("miniaero_mapper.h",
                                          { "-I", root_dir,
                                            "-I", runtime_dir,
                                            "-I", mapper_dir })

  if not saveobj then
    terralib.linklibrary(mapper_binary)
    mapper_header.register_mappers()
    regentlib.start(toplevel)
  else
    local root_dir = arg[0]:match(".*/") or "./"
    local link_flags = terralib.newlist({"-L" .. root_dir, "-l" .. prefix, "-lm"})
    if os.getenv('CRAYPE_VERSION') then
      local new_flags = terralib.newlist({"-Wl,-Bdynamic"})
      new_flags:insertall(link_flags)
      for flag in os.getenv('CRAY_UGNI_POST_LINK_OPTS'):gmatch("%S+") do
        new_flags:insert(flag)
      end
      new_flags:insert("-lugni")
      for flag in os.getenv('CRAY_UDREG_POST_LINK_OPTS'):gmatch("%S+") do
        new_flags:insert(flag)
      end
      new_flags:insert("-ludreg")
      link_flags = new_flags
    end

    regentlib.saveobj(toplevel, "miniaero_" .. prefix, "executable",
        mapper_header.register_mappers, link_flags)
  end
end

return launcher
