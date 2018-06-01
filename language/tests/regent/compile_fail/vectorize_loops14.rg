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

-- fails-with:
-- vectorize_loops14.rg:45: vectorization failed: loop body has aliasing update of path region(fs3()).v
--     e.p1.p.v += e.p2.p.v
--     ^

import "regent"

fspace fs3
{
  v : float,
}

fspace fs2(r : region(fs3))
{
  p : ptr(fs3, r),
}

fspace fs1(s : region(fs3), r : region(fs2(s)))
{
  p1 : ptr(fs2(s), r),
  p2 : ptr(fs2(s), r),
}

task f(s : region(fs3), r : region(fs2(s)), t : region(fs1(s, r)))
where
  reads(s.v, r.p, t.p1, t.p2),
  writes(s.v)
do
  __demand(__vectorize)
  for e in t do
    e.p1.p.v += e.p2.p.v
  end
end

-- FIXME: This test was supposed to check this case. Put this back once the vectorizer gets fixed.
-- vectorize_loops14.rg:45: vectorization failed: loop body has aliasing update of path region(fs3()).v
--    e.p1.p.v += e.p2.p.v
--    ^
