-- Copyright 2024 Stanford University
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

fspace tree(rtop : region(tree(wild))) {
  rleft : region(tree(wild)),
  rright : region(tree(wild)),
  left : ptr(tree(rleft), rleft),
  right : ptr(tree(rright), rright),
}
where
  rleft <= rtop,
  rright <= rtop,
  rleft * rright
end

task f(rtop : region(tree(rtop)), t : ptr(tree(rtop), rtop))
where reads writes(rtop) do
end

task main()
  var r = region(ispace(ptr, 4), tree(r))
  var x = dynamic_cast(ptr(tree(r), r), 0)
  f(r, x)
end
regentlib.start(main)
