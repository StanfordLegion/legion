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

-- This file is not meant to be run directly.

-- runs-with:
-- []

-- Lua/Terra implementation of DE Shaw's Philox 2x32 PRNG
-- (http://dx.doi.org/10.1145/2063384.2063405)

-- helper function to just stamp out n copies of some terra block
local repblk = function(n, blk)
  local rblk = blk
  for i = 2,n do
    rblk = quote [ rblk ] blk end
  end
  return rblk
end

-- generator function to produce a PRNG from the 2x32 family with the
--  specified number of rounds - the recommended number is 10, but
--  ~5 can be ok if you need something fast
local philox2x32 = function(rounds)
  local q = terra(key : uint32, ctr : uint64) : uint64
    var ctr_lo : uint32
    var ctr_hi : uint32
    var prod_lo : uint32
    var prod_hi : uint32
    ctr_lo = ctr
    ctr_hi = ctr >> 32
    [ repblk(10, quote
      var prod : uint64 = ctr_lo * 0xD256D193ULL;
      prod_hi = prod >> 32
      prod_lo = prod
      ctr_lo = ctr_hi ^ key ^ prod_hi
      ctr_hi = prod_lo
      key = key + 0x9E3779B9U
      end) ]
    return ([uint64](ctr_hi) << 32) + ctr_lo
  end
  return q
end

return { philox2x32 = philox2x32,
	 u64tofp = terra(x : uint64) : double return 0x1p-64 * x end
         }

