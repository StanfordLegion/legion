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
-- []

import "regent"

local c = regentlib.c

fspace tree(rtop : region(tree(wild))) {
  rleft : region(tree(wild)),
  rright : region(tree(wild)),
  left : ptr(tree(rleft), rleft),
  right : ptr(tree(rright), rright),
  data : int,
} where rleft <= rtop, rright <= rtop, rleft * rright end

task sum_tree(rtop : region(tree(wild)), node : ptr(tree(rtop), rtop)) : int
where
  reads(rtop)
do

  if isnull(node) then
    return 0
  end

  var {rleft, rright, left, right, data } = @node
  return data + sum_tree(rleft, left) + sum_tree(rright, right)
end

task reverse_tree(rtop : region(tree(wild)), node : ptr(tree(rtop), rtop))
where
  reads(rtop), writes(rtop)
do

  if not isnull(node) then
    var {rleft, rright, left, right, data } = @node
    reverse_tree(rleft, left)
    reverse_tree(rright, right)
    @node = [tree(rtop)]{rleft = rright, rright = rleft, left = right, right = left, data = data}
  end
end

task make_tree(rtop : region(tree(wild)), low : int, high : int)
  : ptr(tree(rtop), rtop)
where
  reads(rtop), writes(rtop)
do

  if low > high then
    return null(ptr(tree(rtop), rtop))
  end

  var mid = (high + low)/2
  var colors = [c.legion_coloring_create]();
  [c.legion_coloring_ensure_color](colors, 0);
  [c.legion_coloring_ensure_color](colors, 1)
  var part = partition(disjoint, rtop, colors)
  [c.legion_coloring_destroy](colors)
  var rleft = part[0]
  var rright = part[1]
  var left = make_tree(rleft, low, mid - 1)
  var right = make_tree(rright, mid + 1, high)
  var top = new(ptr(tree(rtop), rtop))
  @top = [tree(rtop)] { rleft = rleft, rright = rright, left = left, right = right, data = mid }
  return top
end

task top() : int
  var rtop = region(ispace(ptr, 16), tree(wild))
  var root = make_tree(rtop, 0, 15)
  var s1 = sum_tree(rtop, root)
  reverse_tree(rtop, root)
  var s2 = sum_tree(rtop, root)
  if s1 == s2 then
    return s2
  else
    return 0
  end
end

task main()
  assert(top() == 120)
end
regentlib.start(main)
