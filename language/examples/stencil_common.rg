-- Copyright 2018 Stanford University
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
-- []

local common = {}

local c = regentlib.c
local cstring = terralib.includec("string.h")

local config_fields = terralib.newlist({
  -- Command-line parameters.
  {field = "nx", type = int64, default_value = 12},
  {field = "ny", type = int64, default_value = 12},
  {field = "ntx", type = int64, default_value = 4},
  {field = "nty", type = int64, default_value = 4},
  {field = "tsteps", type = int64, default_value = 20},
  {field = "tprune", type = int64, default_value = 5},
  {field = "init", type = int64, default_value = 1000},
})

local config = terralib.types.newstruct("config")
config.entries:insertall(config_fields)

local terra get_optional_arg(key : rawstring)
  var args = c.legion_runtime_get_input_args()
  var i = 1
  while i < args.argc do
    if cstring.strcmp(args.argv[i], key) == 0 then
      if i + 1 < args.argc then
        return args.argv[i + 1]
      else
        return nil
      end
    end
    i = i + 1
  end
  return nil
end

terra common.read_config()
  var conf : config

  -- Set defaults.
  [config_fields:map(function(field)
       return quote conf.[field.field] = [field.default_value] end
     end)]

  var nx = get_optional_arg("-nx")
  if nx ~= nil then
    conf.nx = c.atoll(nx)
  end
  if conf.nx <= 0 then
    c.printf("Error: nx (%lld) must be > 0\n", conf.nx)
    c.abort()
  end

  var ny = get_optional_arg("-ny")
  if ny ~= nil then
    conf.ny = c.atoll(ny)
  end
  if conf.ny <= 0 then
    c.printf("Error: ny (%lld) must be > 0\n", conf.ny)
    c.abort()
  end

  var ntx = get_optional_arg("-ntx")
  if ntx ~= nil then
    conf.ntx = c.atoll(ntx)
  end
  if conf.ntx <= 0 then
    c.printf("Error: ntx (%lld) must be > 0\n", conf.ntx)
    c.abort()
  end

  var nty = get_optional_arg("-nty")
  if nty ~= nil then
    conf.nty = c.atoll(nty)
  end
  if conf.nty <= 0 then
    c.printf("Error: nty (%lld) must be > 0\n", conf.nty)
    c.abort()
  end

  var tsteps = get_optional_arg("-tsteps")
  if tsteps ~= nil then
    conf.tsteps = c.atoll(tsteps)
  end
  if conf.tsteps <= 0 then
    c.printf("Error: tsteps (%lld) must be > 0\n", conf.tsteps)
    c.abort()
  end

  var tprune = get_optional_arg("-tprune")
  if tprune ~= nil then
    conf.tprune = c.atoll(tprune)
  end
  if conf.tprune <= 0 then
    c.printf("Error: tprune (%lld) must be > 0\n", conf.tprune)
    c.abort()
  end

  var init = get_optional_arg("-init")
  if init ~= nil then
    conf.init = c.atoll(init)
  end
  if conf.init <= 0 then
    c.printf("Error: init (%lld) must be > 0\n", conf.init)
    c.abort()
  end

  return conf
end

return common
