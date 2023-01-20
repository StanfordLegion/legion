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

-- runs-with:
-- [
--  ["-ll:cpu", "2" ],
--  ["-ll:cpu", "2", "-ffuture", "0"]
-- ]

import "regent"

task foo()
  return 1
end

task bar(x : int)
end

task toplevel()
  var x = foo()
  must_epoch
    bar(x)
    bar(x)
  end
end

regentlib.start(toplevel)
