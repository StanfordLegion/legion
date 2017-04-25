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

local omp_abi = terralib.includecstring [[
extern int omp_get_num_threads(void);
extern int omp_get_thread_num(void);
extern void GOMP_parallel(void (*fnptr)(void *data), void *data, int nthreads, unsigned flags);
]]

local omp = {
  get_num_threads = omp_abi.omp_get_num_threads,
  get_thread_num = omp_abi.omp_get_thread_num,
  launch = omp_abi.GOMP_parallel,
}

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

return omp
