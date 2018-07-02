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

import "regent"

local struct Particle {
  __valid : bool;
}

__demand(__parallel)
task particles_initValidField(r : region(ispace(int1d), Particle))
where writes(r.__valid) do
  for e in r do
    e.__valid = false
  end
end

local particles = regentlib.newsymbol()
local p_particles = regentlib.newsymbol()

local DeclSymbols = rquote
  var [particles] = region(ispace(int1d,32), Particle)
  var primColors = ispace(int1d, 2)
  var [p_particles] = partition(equal, particles, primColors)
end

local InitRegions = rquote
  __parallelize_with [p_particles] do
    particles_initValidField(particles)
  end
end

local task main()
  [DeclSymbols];
  [InitRegions];
end

regentlib.start(main)
