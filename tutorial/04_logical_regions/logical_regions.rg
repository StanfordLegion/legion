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

-- A field space (fspace) is a collection of fields, similar to a
-- C struct.
fspace fs {
  a : double,
  {b, c, d} : int, -- Multiple fields may be declared with a single type.
}

task main()
  -- An index space (ispace) is a collection in index points. Regent
  -- has two kinds of index spaces: structured and unstructured.

  -- An unstructured ispace is a collection of opaque points, useful
  -- for pointer data structures such as graphs, trees, linked lists,
  -- and unstructured meshes. The following line creates an ispace
  -- with 1024 elements.
  var unstructured_is = ispace(ptr, 1024)

  -- A structured ispace is (multi-dimensional) rectangle of
  -- points. The space below includes the 1-dimensional ints from 0 to 1023.
  var structured_is = ispace(int1d, 1024, 0)

  -- A region is the cross product between an ispace and an fspace.
  var unstructured_lr = region(unstructured_is, fs)
  var structured_lr = region(structured_is, fs)

  -- Note that you can create multiple regions with the same ispace
  -- and fspace. This is a **NEW** region, distint from structured_lr
  -- above.
  var no_clone_lr = region(structured_is, fs)
end
regentlib.start(main)
