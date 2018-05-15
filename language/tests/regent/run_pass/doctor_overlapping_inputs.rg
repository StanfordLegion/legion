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

local c = regentlib.c

fspace p1 {
	a : int,
	b : int,
}

-- Reads for int regions, ok.
task both_reads(r : region(int), s : region(int))
where reads(r, s) do
end

-- Read/write for int regions, complain.
task read_write(r : region(int), s : region(int))
where reads(r), writes(s) do
end

-- Same reduction for int regions, ok.
task same_reduction(r : region(int), s : region(int))
where reduces max(r, s) do
end

-- Different reductions for int regions, complain.
task diff_reduction(r : region(int), s : region(int))
where reduces max(r), reduces min(s) do
end

-- Read/write for custom regions, complain.
task both_rw(r : region(p1), s : region(p1))
where reads(r, s), writes(r, s) do
end

task main()
	var ti = region(ispace(ptr, 5), int)
	var tp = region(ispace(ptr, 5), p1)

	var colors = c.legion_coloring_create()
	c.legion_coloring_ensure_color(colors, 0)
	c.legion_coloring_ensure_color(colors, 1)

	var tip = partition(disjoint, ti, colors)
	var ti0 = tip[0]
	var ti1 = tip[1]

	both_reads(ti, ti0) --ok
	read_write(ti, ti1) --bad
	same_reduction(ti, ti0) --ok
	diff_reduction(ti, ti1) --bad

	var tpp = partition(disjoint, tp, colors)
	var tp0 = tpp[0]
	var tp1 = tpp[1]

	both_rw(tp, tp0) --bad

	c.legion_coloring_destroy(colors)
end

regentlib.start(main)