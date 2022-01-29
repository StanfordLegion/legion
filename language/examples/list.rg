-- Copyright 2022 Stanford University
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

fspace list(a : region(list(a))) {
  data : int,
  next : ptr(list(a), a),
}

task make_list(d : region(list(d)), n : int) : ptr(list(d), d)
where
  reads(d), writes(d)
do
  var x = null(ptr(list(d), d))
  var i = n
  for y in d do
    @y = [list(d)]{ data = i, next = x }
    x = y
    i -= 1
  end
  return x
end

task sum_list(e : region(list(e)), x : ptr(list(e), e)) : int
where
  reads(e), writes(e)
do
  var s = 0
  while not isnull(x) do
    s = s + x.data
    x = x.next
  end
  return s
end

task top() : int
  var f = region(ispace(ptr, 5), list(f))
  var x = make_list(f, 5)
  return sum_list(f, x)
end

task main()
  regentlib.assert(top() == 15, "test failed")
end
regentlib.start(main)
