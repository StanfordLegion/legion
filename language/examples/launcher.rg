local launcher = {}

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

function launcher.compile_mapper(saveobj, name)
  local cc_file = root_dir .. name .. ".cc"
  local binary_file

  if saveobj then
    binary_file = root_dir .. "lib" .. name .. ".so"
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
  local max_dim = os.getenv('MAX_DIM') or '3'
  cxx_flags = cxx_flags .. " -DLEGION_MAX_DIM=" .. max_dim .. " -DREALM_MAX_DIM=" .. max_dim

  local cmd = (cxx .. " " .. cxx_flags .. " " .. include_path .. " " ..
               cc_file .. " -o " .. binary_file)
  if os.execute(cmd) ~= 0 then
    print("Error: failed to compile " .. cc_file)
    assert(false)
  end
  return binary_file
end

function launcher.launch(toplevel, name)
  local saveobj = os.getenv('SAVEOBJ') == '1'
  local mapper_binary = launcher.compile_mapper(saveobj, name)
  local mapper_header = terralib.includec("miniaero_mapper.h", include_dirs)

  if not saveobj then
    terralib.linklibrary(mapper_binary)
    mapper_header.register_mappers()
    regentlib.start(toplevel)
  else
    local root_dir = arg[0]:match(".*/") or "./"
    local link_flags = terralib.newlist({"-L" .. root_dir, "-l" .. name, "-lm"})
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

    regentlib.saveobj(toplevel, name, "executable",
        mapper_header.register_mappers, link_flags)
  end
end

return launcher
