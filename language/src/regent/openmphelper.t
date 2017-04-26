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

local config = require("regent/config")

local omp = {}

local has_openmp = false
do
  local dlfcn = terralib.includec("dlfcn.h")
  local terra find_openmp_symbols()
    var lib = dlfcn.dlopen([&int8](0), dlfcn.RTLD_LAZY)
    var has_openmp = 
      dlfcn.dlsym(lib, "GOMP_parallel") ~= [&opaque](0) and
      dlfcn.dlsym(lib, "omp_get_num_threads") ~= [&opaque](0) and
      dlfcn.dlsym(lib, "omp_get_thread_num") ~= [&opaque](0)
    dlfcn.dlclose(lib)
    return has_openmp
  end
  has_openmp = find_openmp_symbols()
end

if not (config.args()["openmp"] and has_openmp) then
  omp.check_openmp_available = function() return false end
  terra omp.get_num_threads() return 1 end
  terra omp.get_thread_num() return 0 end
  local omp_worker_type =
    terralib.types.functype(terralib.newlist({&opaque}), terralib.types.unit, false)
  terra omp.launch(fnptr : &omp_worker_type, data : &opaque, nthreads : int32, flags : uint32)
    fnptr(data)
  end
else
  omp.check_openmp_available = function() return true end
  local omp_abi = terralib.includecstring [[
    extern int omp_get_num_threads(void);
    extern int omp_get_thread_num(void);
    extern void GOMP_parallel(void (*fnptr)(void *data), void *data, int nthreads, unsigned flags);
  ]]

  omp.get_num_threads = omp_abi.omp_get_num_threads
  omp.get_thread_num = omp_abi.omp_get_thread_num
  omp.launch = omp_abi.GOMP_parallel
end

function omp.generate_preamble_structured(rect, idx, start_idx, end_idx)
  return quote
    var num_threads = [omp.get_num_threads]()
    var thread_id = [omp.get_thread_num]()
    var lo = [rect].lo.x[idx]
    var hi = [rect].hi.x[idx] + 1
    var chunk = (hi - lo + num_threads - 1) / num_threads
    if chunk == 0 then chunk = 1 end
    var [start_idx] = thread_id * chunk + lo
    var [end_idx] = (thread_id + 1) * chunk + lo
    if [end_idx] > hi then [end_idx] = hi end
  end
end

function omp.generate_argument_type(symbols)
  local arg_type = terralib.types.newstruct("omp_worker_arg")
  arg_type.entries = terralib.newlist()
  local mapping = {}
  for i, symbol in pairs(symbols) do
    local field_name = "_arg" .. tostring(i)
    arg_type.entries:insert({ field_name, symbol.type })
    mapping[field_name] = symbol
  end
  return arg_type, mapping
end

function omp.generate_argument_init(arg, arg_type, mapping)
  local worker_init = arg_type.entries:map(function(pair)
    local field_name, field_type = unpack(pair)
    return quote var [ mapping[field_name] ] = [arg].[field_name] end
  end)
  local launch_init = arg_type.entries:map(function(pair)
    local field_name, field_type = unpack(pair)
    return quote [arg].[field_name] = [ mapping[field_name] ] end
  end)
  return worker_init, launch_init
end

return omp
