/*
 * Copyright 2025 Stanford University, NVIDIA Corporation
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "realm/transfer/transfer.h"
#include "realm/transfer/ib_memory.h"
#include <gtest/gtest.h>
#include <limits>
#include <map>

using namespace Realm;

namespace Realm {
  void redistribute_ib_sizes_impl(std::vector<TransferGraph::IBInfo> &ib_edges,
                                  const std::map<Memory, size_t> &caps);
  size_t compute_ib_min_size(size_t combined_field_size, CustomSerdezID serdez_id,
                             size_t max_needed);
  size_t compute_ib_natural_size(size_t combined_field_size, size_t domain_size,
                                 CustomSerdezID serdez_id);
} // namespace Realm

namespace {

  constexpr size_t A = IB_ALLOC_ALIGNMENT; // 256

  TransferGraph::IBInfo mk(Memory memory, size_t size, size_t min_size,
                           size_t size_granularity = 1)
  {
    TransferGraph::IBInfo e;
    e.memory = memory;
    e.size = size;
    e.min_size = min_size;
    e.size_granularity = size_granularity;
    return e;
  }

  // total bytes the allocator would consume for all edges targeting mem
  size_t rounded_total(const std::vector<TransferGraph::IBInfo> &edges, Memory mem)
  {
    size_t sum = 0;
    for(const auto &e : edges) {
      if(e.memory == mem) {
        sum += ib_align_up(e.size);
      }
    }
    return sum;
  }

} // namespace

// Edges that already fit are left untouched.
TEST(IBRedistributeTest, LeavesFittingPlanUntouched)
{
  const Memory M(0x1000);
  std::vector<TransferGraph::IBInfo> edges = {
      mk(M, 100 * A, 4 * A),
      mk(M, 100 * A, 4 * A),
  };
  std::map<Memory, size_t> caps = {{M, 1024 * A}};

  redistribute_ib_sizes_impl(edges, caps);

  EXPECT_EQ(edges[0].size, 100u * A);
  EXPECT_EQ(edges[1].size, 100u * A);
}

// Oversubscribed same-memory edges are shrunk so the rounded aggregate fits,
// each stays 256-aligned and >= its floor, and the larger requester keeps the
// larger share.
TEST(IBRedistributeTest, ShrinksProportionallyToFit)
{
  const Memory M(0x2000);
  const size_t cap = 1024 * A;
  std::vector<TransferGraph::IBInfo> edges = {
      mk(M, 300 * A, 10 * A), // small requester
      mk(M, 900 * A, 10 * A), // large requester
  };
  std::map<Memory, size_t> caps = {{M, cap}};

  redistribute_ib_sizes_impl(edges, caps);

  EXPECT_LE(rounded_total(edges, M), cap);
  for(const auto &e : edges) {
    EXPECT_EQ(e.size % A, 0u) << "size must stay 256-aligned";
    EXPECT_GE(e.size, 10u * A) << "size must not drop below its floor";
    EXPECT_LE(e.size, ib_align_up(e.size)) << "sanity";
  }
  // the edge that asked for more keeps more
  EXPECT_GT(edges[1].size, edges[0].size);
  // and neither is grown beyond what it requested
  EXPECT_LE(edges[0].size, 300u * A);
  EXPECT_LE(edges[1].size, 900u * A);
}

// A single edge larger than its memory is shrunk to fit rather than aborting.
TEST(IBRedistributeTest, ShrinksSingleOversizedEdge)
{
  const Memory M(0x3000);
  const size_t cap = 512 * A;
  std::vector<TransferGraph::IBInfo> edges = {mk(M, 4096 * A, 8 * A)};
  std::map<Memory, size_t> caps = {{M, cap}};

  redistribute_ib_sizes_impl(edges, caps);

  EXPECT_LE(ib_align_up(edges[0].size), cap);
  EXPECT_GE(edges[0].size, 8u * A);
  EXPECT_EQ(edges[0].size % A, 0u);
}

// Metadata edges with no explicit floor (min_size == 0) are never shrunk below
// the 256-byte alignment.
TEST(IBRedistributeTest, RespectsDefaultFloorForMetadataEdges)
{
  const Memory M(0x4000);
  const size_t cap = 300 * A;
  std::vector<TransferGraph::IBInfo> edges = {
      mk(M, 256 * A, 0),      // metadata edge, no explicit floor
      mk(M, 256 * A, 32 * A), // real edge
  };
  std::map<Memory, size_t> caps = {{M, cap}};

  redistribute_ib_sizes_impl(edges, caps);

  EXPECT_LE(rounded_total(edges, M), cap);
  EXPECT_GE(edges[0].size, A) << "metadata edge must not drop below 256";
  EXPECT_GE(edges[1].size, 32u * A) << "real edge must keep its floor";
}

// Different memories are handled independently.
TEST(IBRedistributeTest, HandlesMultipleMemoriesIndependently)
{
  const Memory M1(0x5000), M2(0x6000);
  std::vector<TransferGraph::IBInfo> edges = {
      mk(M1, 900 * A, 8 * A), // M1 oversubscribed with the next edge
      mk(M2, 10 * A, 2 * A),  // M2 fits comfortably, must be untouched
      mk(M1, 900 * A, 8 * A),
  };
  std::map<Memory, size_t> caps = {{M1, 1024 * A}, {M2, 1024 * A}};

  redistribute_ib_sizes_impl(edges, caps);

  EXPECT_LE(rounded_total(edges, M1), 1024u * A);
  EXPECT_EQ(edges[1].size, 10u * A) << "fitting memory must be left untouched";
}

// Shrunk payload edges keep their transfer granularity, not just allocator
// alignment.
TEST(IBRedistributeTest, PreservesEdgeGranularityWhenShrinking)
{
  const Memory M(0x6500);
  const size_t G = 3 * A; // valid transfer unit larger than allocator alignment
  const size_t cap = 9 * G;
  std::vector<TransferGraph::IBInfo> edges = {
      mk(M, 6 * G, A, G),
      mk(M, 6 * G, A, G),
  };
  std::map<Memory, size_t> caps = {{M, cap}};

  redistribute_ib_sizes_impl(edges, caps);

  EXPECT_LE(rounded_total(edges, M), cap);
  for(const auto &e : edges) {
    EXPECT_EQ(e.size % G, 0u);
    EXPECT_GE(e.size, G);
    EXPECT_LE(e.size, 6u * G);
  }
}

// The progress floor is the raw 2-element ring size (NOT rounded up to the
// allocator's 256-byte alignment), and it is capped at the transfer's total
// data so small transfers are not spuriously rejected.
TEST(IBMinSizeTest, FloorIsTwoElementsCappedAtDataSize)
{
  constexpr size_t no_cap = std::numeric_limits<size_t>::max();
  // large domain of a small field: floor is 2 elements, not rounded up to 256
  EXPECT_EQ(compute_ib_min_size(4, 0, no_cap), 8u);
  // total data below 2 elements: floor is capped at the data size
  EXPECT_EQ(compute_ib_min_size(4, 0, 4), 4u);
  // a field whose 2-element floor exceeds the total data is capped at the data
  EXPECT_LE(compute_ib_min_size(64, 0, 100), 100u);
  EXPECT_EQ(compute_ib_min_size(64, 0, no_cap), 128u);
}

TEST(IBMinSizeTest, NaturalSizeSaturatesOnOverflow)
{
  constexpr size_t max_size = std::numeric_limits<size_t>::max();
  const size_t large_element = (max_size / 2) + 1;

  EXPECT_EQ(compute_ib_natural_size(large_element, 3, 0), max_size);
  EXPECT_EQ(compute_ib_min_size(large_element, 0, max_size), max_size);
}

TEST(IBMinSizeTest, SmallIndirectFloorsAreCappedAtNaturalSize)
{
  const size_t one_address = compute_ib_natural_size(64, 1, 0);
  const size_t one_payload = compute_ib_natural_size(128, 1, 0);

  EXPECT_EQ(compute_ib_min_size(64, 0, one_address), 64u);
  EXPECT_EQ(compute_ib_min_size(128, 0, one_payload), 128u);
  EXPECT_EQ(compute_ib_min_size(sizeof(unsigned), 0, std::numeric_limits<size_t>::max()),
            2u * sizeof(unsigned));
}

// Regression: a small multi-hop transfer whose logical ring size is below the
// allocator's 256-byte alignment but at/above its own progress floor must be
// preserved, not rejected.  This mirrors init_transfer_ib_edge's output for a
// 10-element x 4-byte copy: size=40, min_size=min(2*4, 40)=8, granularity=4.
TEST(IBRedistributeTest, PreservesSmallSubAlignmentEdge)
{
  const Memory M(0x8000);
  std::vector<TransferGraph::IBInfo> edges = {mk(M, 40, 8, 4)};
  std::map<Memory, size_t> caps = {{M, 128ull * 1024 * 1024}};
  redistribute_ib_sizes_impl(edges, caps);
  EXPECT_EQ(edges[0].size, 40u) << "small transfer must not be rejected or shrunk";
}

// A hard progress floor is enforced even when the aggregate allocation would
// fit in the target memory.
TEST(IBRedistributeDeathTest, AbortsWhenEdgeBelowHardFloor)
{
  const Memory M(0x6800);
  std::vector<TransferGraph::IBInfo> edges = {mk(M, 2 * A, 4 * A)};
  std::map<Memory, size_t> caps = {{M, 1024 * A}};

  EXPECT_DEATH(redistribute_ib_sizes_impl(edges, caps), ".*");
}

TEST(IBRedistributeDeathTest, AbortsWhenGranularityLcmOverflows)
{
  const Memory M(0x7800);
  const size_t huge_odd_granularity = std::numeric_limits<size_t>::max() / 2;
  std::vector<TransferGraph::IBInfo> edges = {
      mk(M, 2 * 1024 * A, A, huge_odd_granularity),
  };
  std::map<Memory, size_t> caps = {{M, 1024 * A}};

  EXPECT_DEATH(redistribute_ib_sizes_impl(edges, caps), ".*");
}

TEST(IBRedistributeDeathTest, AbortsWhenAlignedFloorOverflows)
{
  const Memory M(0x7900);
  const size_t max_size = std::numeric_limits<size_t>::max();
  std::vector<TransferGraph::IBInfo> edges = {
      mk(M, max_size, max_size - 1, A),
  };
  std::map<Memory, size_t> caps = {{M, 1024 * A}};

  EXPECT_DEATH(redistribute_ib_sizes_impl(edges, caps), ".*");
}

// When not even one element per edge fits, the plan is genuinely impossible.
TEST(IBRedistributeDeathTest, AbortsWhenFloorsCannotFit)
{
  const Memory M(0x7000);
  const size_t cap = 8 * A; // only 8 units available
  std::vector<TransferGraph::IBInfo> edges = {
      mk(M, 100 * A, 8 * A), // floor 8 units
      mk(M, 100 * A, 8 * A), // floor 8 units -> 16 > 8 avail
  };
  std::map<Memory, size_t> caps = {{M, cap}};

  // the fatal log message is routed through Realm's logger, which isn't
  // initialized in this bare unit test, so just assert the process aborts
  EXPECT_DEATH(redistribute_ib_sizes_impl(edges, caps), ".*");
}
