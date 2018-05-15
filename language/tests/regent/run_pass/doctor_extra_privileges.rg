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

fspace p1 {
  a : int,
  b : int,
}

-- Extra reads, no operation.
task extra_reads_noop(r : region(p1), x : ptr(p1, r)) : int
where
	reads(r.{a, b})
do
end

-- Extra writes, no operation.
task extra_writes_noop(r : region(p1), x : ptr(p1, r)) : int
where
	writes(r.{a, b})
do
end

-- TODO Extra reads, one used in assignment.
task extra_read_assn(r : region(p1), x : ptr(p1, r)) : int
end

-- Extra writes, one used in assignment.
task extra_write_assn(r : region(p1), x : ptr(p1, r)) : int
where
	reads(r.{}), writes(r.{a, b})
do
	x.a = 1
end

-- Extra read in the return.
task extra_read_return(r : region(p1), x : ptr(p1, r)) : int
where reads(r.{a, b}) do
	return x.a
end

-- Extra reduce+.
task extra_plus(r : region(p1), x : ptr(p1, r))
where reduces+(r) do
	x.a += 2
end

-- Extra reduce-.
task extra_minus(r : region(p1), x : ptr(p1, r))
where reduces-(r) do
	x.a -= 2
end

-- Extra reduce*.
task extra_multiply(r : region(p1), x : ptr(p1, r))
where reduces*(r) do
	x.a *= 2
end

-- Extra reduce/.
task extra_divide(r : region(p1), x : ptr(p1, r))
where reduces/(r) do
	x.a /= 2
end

-- Extra reduce min.
task extra_min(r : region(p1), x : ptr(p1, r))
where reduces min(r) do
	x.a min= 2
end

-- Extra reduce max.
task extra_max(r : region(p1), x : ptr(p1, r))
where reduces max(r) do
	x.a max= 2
end

-- Reduce+ with read/write.
-- Expected: Nothing extra
task rw_plus(r : region(p1), x : ptr(p1, r))
where reads(r), writes(r) do
	x.a += 2
end

task check_priv_promotion(r : region(p1), x : ptr(p1, r))
where reads(r), reduces-(r) do
	x.a -= 2
end
