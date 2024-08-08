#include "realm/mem_impl.h"

#include <tuple>
#include <gtest/gtest.h>

using namespace Realm;

class RangeAllocatorTest : public ::testing::Test {
protected:
  const unsigned int SENTINEL = BasicRangeAllocator<size_t, int>::SENTINEL;
  BasicRangeAllocator<size_t, int> range_alloc;
};

TEST_F(RangeAllocatorTest, DeallocateNonExistent)
{
  range_alloc.deallocate(42, /*missiok_ok=*/true);
}

TEST_F(RangeAllocatorTest, DeallocateNonExistentFail)
{
  // TODO(apryakhin): convert to bool return status
  EXPECT_DEATH({ range_alloc.deallocate(42, /*missiok_ok=*/false); }, "");
}

TEST_F(RangeAllocatorTest, LookupEmptyAllocator)
{
  size_t start = 0, size = 0;
  EXPECT_FALSE(range_alloc.lookup(0, start, size));
}

TEST_F(RangeAllocatorTest, AddRange) { range_alloc.add_range(0, 1024); }

TEST_F(RangeAllocatorTest, AddSingleRangeEmpty) { range_alloc.add_range(1024, 1023); }

TEST_F(RangeAllocatorTest, AddMultipleRanges)
{
  range_alloc.add_range(0, 1024);
  // TODO(apryakhin): convert to bool return status
  EXPECT_DEATH({ range_alloc.add_range(1025, 2048); }, "Assertion `0' failed");
}

TEST_F(RangeAllocatorTest, Allocate)
{
  const int range_tag = 42;
  const size_t range_size = 1024;
  const size_t range_align = 16;
  size_t offset = 0;

  range_alloc.add_range(0, range_size);
  bool ok = range_alloc.allocate(range_tag, range_size, range_align, offset);

  EXPECT_TRUE(ok);
  EXPECT_EQ(offset, 0);
}

TEST_F(RangeAllocatorTest, AllocateTooLarge)
{
  const int range_tag = 42;
  const size_t range_size = 1024;
  const size_t range_align = 16;
  size_t offset = 0;

  range_alloc.add_range(0, range_size);
  bool ok = range_alloc.allocate(range_tag, range_size * 2, range_align, offset);

  EXPECT_FALSE(ok);
  EXPECT_EQ(offset, 0);
}

TEST_F(RangeAllocatorTest, AllocateAndLookupInvalidRange)
{
  const int range_tag = 42;
  size_t offset = 0;
  size_t start = 0, size = 0;

  range_alloc.add_range(0, 1024);
  bool alloc_ok = range_alloc.allocate(range_tag, 512, 16, offset);
  bool lookup_ok = range_alloc.lookup(range_tag - 1, start, size);

  EXPECT_TRUE(alloc_ok);
  EXPECT_FALSE(lookup_ok);
}

TEST_F(RangeAllocatorTest, AllocateAndLookupSingleRange)
{
  const int range_tag = 42;
  const size_t range_size = 1024;
  const size_t range_align = 16;
  size_t offset = 0;
  size_t start = 0, size = 0;

  range_alloc.add_range(0, range_size);
  bool alloc_ok = range_alloc.allocate(range_tag, range_size, range_align, offset);
  bool lookup_ok = range_alloc.lookup(range_tag, start, size);

  EXPECT_TRUE(alloc_ok);
  EXPECT_TRUE(lookup_ok);
  EXPECT_EQ(start, 0);
  EXPECT_EQ(size, range_size);
}

TEST_F(RangeAllocatorTest, TestExplicitRangesWithCycle)
{
  {
    range_alloc.ranges[SENTINEL].prev_free = 3;
    range_alloc.ranges[SENTINEL].next_free = 1;
  }
  {
    BasicRangeAllocator<size_t, int>::Range range;
    range.prev_free = SENTINEL;
    range.next_free = 2;
    range_alloc.ranges.push_back(range);
  }
  {
    BasicRangeAllocator<size_t, int>::Range range;
    range.prev_free = 1;
    range.next_free = 3;
    range_alloc.ranges.push_back(range);
  }
  {
    BasicRangeAllocator<size_t, int>::Range range;
    // Link up with range 1
    range.prev_free = 2;
    range.next_free = 1;
    range_alloc.ranges.push_back(range);
  }

  EXPECT_TRUE(range_alloc.free_list_has_cycle());
}

TEST_F(RangeAllocatorTest, TestExplicitRangesNoCycle)
{
  {
    range_alloc.ranges[SENTINEL].prev_free = 3;
    range_alloc.ranges[SENTINEL].next_free = 1;
  }
  {
    BasicRangeAllocator<size_t, int>::Range range;
    range.prev_free = SENTINEL;
    range.next_free = 2;
    range_alloc.ranges.push_back(range);
  }
  {
    BasicRangeAllocator<size_t, int>::Range range;
    range.prev_free = 1;
    range.next_free = 3;
    range_alloc.ranges.push_back(range);
  }
  {
    BasicRangeAllocator<size_t, int>::Range range;
    range.prev_free = 2;
    range.next_free = SENTINEL;
    range_alloc.ranges.push_back(range);
  }

  EXPECT_FALSE(range_alloc.free_list_has_cycle());
}

TEST_F(RangeAllocatorTest, TestExplicitRangesOverlapping)
{
  {
    range_alloc.ranges[SENTINEL].prev = 3;
    range_alloc.ranges[SENTINEL].next = 1;
  }
  {
    BasicRangeAllocator<size_t, int>::Range range;
    range.prev = SENTINEL;
    range.next = 2;
    range.first = 0;
    range.last = 16;
    range_alloc.ranges.push_back(range);
    range_alloc.allocated[1] = 1;
  }
  {
    BasicRangeAllocator<size_t, int>::Range range;
    range.prev = 1;
    range.next = 3;
    range.first = 16;
    range.last = 32;
    range_alloc.ranges.push_back(range);
    range_alloc.allocated[2] = 2;
  }
  {
    BasicRangeAllocator<size_t, int>::Range range;
    range.prev = 2;
    range.next = SENTINEL;
    range.first = 16;
    range.last = 64;
    range_alloc.ranges.push_back(range);
    range_alloc.allocated[3] = 3;
  }

  EXPECT_TRUE(range_alloc.has_invalid_ranges());
}

TEST_F(RangeAllocatorTest, TestExplicitRangesInvalidSize)
{
  {
    range_alloc.ranges[SENTINEL].prev = 3;
    range_alloc.ranges[SENTINEL].next = 1;
  }
  {
    BasicRangeAllocator<size_t, int>::Range range;
    range.prev = SENTINEL;
    range.next = 2;
    range.first = 0;
    range.last = 16;
    range_alloc.ranges.push_back(range);
    range_alloc.allocated[1] = 1;
  }
  {
    BasicRangeAllocator<size_t, int>::Range range;
    range.prev = 1;
    range.next = 3;
    range.first = 16;
    range.last = 15;
    range_alloc.ranges.push_back(range);
    range_alloc.allocated[2] = 2;
  }
  {
    BasicRangeAllocator<size_t, int>::Range range;
    range.prev = 2;
    range.next = SENTINEL;
    range.first = 32;
    range.last = 64;
    range_alloc.ranges.push_back(range);
    range_alloc.allocated[3] = 3;
  }

  EXPECT_TRUE(range_alloc.has_invalid_ranges());
}

TEST_F(RangeAllocatorTest, TestExplicitRangesValid)
{
  {
    range_alloc.ranges[SENTINEL].prev = 3;
    range_alloc.ranges[SENTINEL].next = 1;
  }
  {
    BasicRangeAllocator<size_t, int>::Range range;
    range.prev = SENTINEL;
    range.next = 2;
    range.first = 0;
    range.last = 16;
    range_alloc.ranges.push_back(range);
    range_alloc.allocated[1] = 1;
  }
  {
    BasicRangeAllocator<size_t, int>::Range range;
    range.prev = 1;
    range.next = 3;
    range.first = 16;
    range.last = 20;
    range_alloc.ranges.push_back(range);
    range_alloc.allocated[2] = 2;
  }
  {
    BasicRangeAllocator<size_t, int>::Range range;
    range.prev = 2;
    range.next = SENTINEL;
    range.first = 32;
    range.last = 64;
    range_alloc.ranges.push_back(range);
    range_alloc.allocated[3] = 3;
  }

  EXPECT_FALSE(range_alloc.has_invalid_ranges());
}

TEST_F(RangeAllocatorTest, TestExplicitRangesSharedTags)
{
  {
    range_alloc.ranges[SENTINEL].prev = 3;
    range_alloc.ranges[SENTINEL].next = 1;
  }
  {
    BasicRangeAllocator<size_t, int>::Range range;
    range.prev = SENTINEL;
    range.next = 2;
    range.first = 0;
    range.last = 16;
    range_alloc.ranges.push_back(range);
    range_alloc.allocated[1] = 1;
  }
  {
    BasicRangeAllocator<size_t, int>::Range range;
    range.prev = 1;
    range.next = 3;
    range.first = 16;
    range.last = 20;
    range_alloc.ranges.push_back(range);
    range_alloc.allocated[2] = 2;
  }
  {
    BasicRangeAllocator<size_t, int>::Range range;
    range.prev = 2;
    range.next = SENTINEL;
    range.first = 32;
    range.last = 64;
    range_alloc.ranges.push_back(range);
    range_alloc.allocated[3] = 2;
  }

  EXPECT_TRUE(range_alloc.has_invalid_ranges());
}

struct RangeAllocTestCase {
  std::vector<std::pair<size_t, size_t>> alloc_ranges;

  std::vector<int> alloc_tags;
  std::vector<size_t> alloc_sizes;
  std::vector<size_t> alloc_aligns;

  // std::vector<int> split_old_tag;
  // std::vector<int> split_status;

  struct SplitOp {
    int old_tag;
    int good_allocs;
    std::vector<int> new_tags;
    // std::vector<bool> split_lookups_exp;
    std::vector<size_t> sizes;
    std::vector<size_t> aligns;
    std::vector<size_t> exp_offsets;
  };

  std::vector<SplitOp> split_ops;
  std::vector<BasicRangeAllocator<size_t, int>::Range> exp_ranges;

  size_t free_size;
};

typedef BasicRangeAllocator<size_t, int>::Range Range;

class RangeAllocatorSplitParamTest : public ::testing::TestWithParam<RangeAllocTestCase> {
protected:
  const unsigned int SENTINEL = BasicRangeAllocator<size_t, int>::SENTINEL;
  size_t get_total_free_size()
  {
    size_t total_free_size = 0;
    const std::vector<Range> &ranges = range_alloc.ranges;
    unsigned free_idx = ranges[SENTINEL].next_free;
    while(free_idx != SENTINEL) {
      total_free_size += (ranges[free_idx].last - ranges[free_idx].first);
      free_idx = ranges[free_idx].next_free;
    }
    return total_free_size;
  }

  BasicRangeAllocator<size_t, int> range_alloc;
};

TEST_P(RangeAllocatorSplitParamTest, Base)
{
  auto test_case = GetParam();

  for(size_t i = 0; i < test_case.alloc_ranges.size(); i++) {
    range_alloc.add_range(test_case.alloc_ranges[i].first,
                          test_case.alloc_ranges[i].second);
  }

  for(size_t i = 0; i < test_case.alloc_tags.size(); i++) {
    size_t offset = 0;
    EXPECT_TRUE(range_alloc.allocate(test_case.alloc_tags[i], test_case.alloc_sizes[i],
                                     test_case.alloc_aligns[i], offset));
  }

  for(RangeAllocTestCase::SplitOp op : test_case.split_ops) {
    std::vector<size_t> offsets(op.new_tags.size());
    EXPECT_EQ(
        range_alloc.split_range(op.old_tag, op.new_tags, op.sizes, op.aligns, offsets),
        op.good_allocs);
    for(size_t j = 0; j < offsets.size(); j++) {
      EXPECT_EQ(offsets[j], op.exp_offsets[j]);
    }
  }

  size_t free_size = get_total_free_size();

  EXPECT_EQ(test_case.free_size, free_size);

  if(!test_case.exp_ranges.empty()) {
    // EXPECT_EQ(range_alloc.ranges.size(), test_case.exp_ranges.size());

    size_t index = 1;
    unsigned idx = range_alloc.ranges[SENTINEL].next;
    while(idx != SENTINEL) {
      EXPECT_EQ(range_alloc.ranges[idx].first, test_case.exp_ranges[index].first)
          << " index:" << index;
      EXPECT_EQ(range_alloc.ranges[idx].last, test_case.exp_ranges[index].last)
          << " index:" << index;
      idx = range_alloc.ranges[idx].next;
      index++;
    }
  }

  EXPECT_FALSE(range_alloc.free_list_has_cycle());
  EXPECT_FALSE(range_alloc.has_invalid_ranges());

  // TODO(apryakhin@)
  // for(size_t i = 0; i < test_case.split_new_tags.size(); i++) {
  // EXPECT_EQ(range_alloc.allocated.find());
  //}
}

INSTANTIATE_TEST_SUITE_P(
    RangeAlloc, RangeAllocatorSplitParamTest,
    testing::Values(

        // Case 0: split empty
        RangeAllocTestCase{.split_ops{{.old_tag = 1,
                                       .good_allocs = 0,
                                       .new_tags{7},
                                       .sizes{512},
                                       .aligns{8},
                                       .exp_offsets{0}}},
                           .exp_ranges{Range{}}},

        // Case 1: split from non-existent tag
        RangeAllocTestCase{.alloc_ranges{{0, 512}},
                           .alloc_tags{1},
                           .alloc_sizes{512},
                           .alloc_aligns{8},

                           .split_ops{{.old_tag = 2,
                                       .good_allocs = 0,
                                       .new_tags{7},
                                       .sizes{512},
                                       .aligns{8},
                                       .exp_offsets{0}}},

                           .exp_ranges{Range{/*first=*/0, /*last=*/0},
                                       Range{/*first=*/0, /*last=*/512}}},

        // TODO(apryakhin@): Consider enabling it back
        // Case 2: split into the existing tag
        /*RangeAllocTestCase{.alloc_ranges{{0, 512}},
                 .alloc_tags{1},
                 .alloc_sizes{512},
                 .alloc_aligns{8},

                 .split_ops{{.old_tag = 1,
                             .good_allocs = 1,
                             .new_tags{1},
                             .sizes{512},
                             .aligns{8},
                             .exp_offsets{0}}},

                 .exp_ranges{Range{0, 0},
                             Range{0, 512}}},*/

        // Case 3: base case split/reuse the full range
        RangeAllocTestCase{.alloc_ranges{{0, 512}},
                           .alloc_tags{1},
                           .alloc_sizes{512},
                           .alloc_aligns{8},

                           .split_ops{{.old_tag = 1,
                                       .good_allocs = 1,
                                       .new_tags{7},
                                       .sizes{512},
                                       .aligns{8},
                                       .exp_offsets{0}}},

                           .exp_ranges{
                               Range{/*first=*/0, /*last=*/0},
                               Range{/*first=*/0, /*last=*/512},
                               Range{/*first=*/512, /*last=*/512},
                           }},

        // Case 4: split/reuse zero range
        RangeAllocTestCase{
            .alloc_ranges{{0, 512}},
            .alloc_tags{1},
            .alloc_sizes{0},
            .alloc_aligns{1},

            .split_ops{{.old_tag = 1,
                        .good_allocs = 1,
                        .new_tags{7},
                        .sizes{0},
                        .aligns{8},
                        .exp_offsets{0}}},

            .exp_ranges{Range{/*first=*/0, /*last=*/0}, Range{/*first=*/0, /*last=*/512}},

            .free_size = 512,
        },

        // Case 5: reuse range with different alignment and create a free block in front
        RangeAllocTestCase{
            .alloc_ranges{{24, 1000}},
            .alloc_tags{1},
            .alloc_sizes{800},
            .alloc_aligns{24},

            .split_ops{{.old_tag = 1,
                        .good_allocs = 1,
                        .new_tags{7},
                        .sizes{496},
                        .aligns{16},
                        .exp_offsets{32}}},

            .exp_ranges{Range{/*first=*/0, /*last=*/0},

                        Range{/*first=*/24, /*last=*/32},

                        Range{/*first=*/32, /*last=*/528},

                        Range{/*first=*/528, /*last=*/1000}},

            .free_size = 480,
        },

        // Case 6: split range with second layout going OOM
        RangeAllocTestCase{
            .alloc_ranges{{24, 1000}},
            .alloc_tags{1},
            .alloc_sizes{800},
            .alloc_aligns{24},

            .split_ops{{.old_tag = 1,
                        .good_allocs = 1,
                        .new_tags{7, 8},
                        .sizes{248, 1024},
                        .aligns{16, 24},
                        .exp_offsets{32, 0}}},

            .exp_ranges{
                Range{/*first=*/0, /*last=*/0},

                // free range after alignment computation
                Range{/*first=*/24, /*last=*/32},

                Range{/*first=*/32, /*last=*/280},

                Range{/*first=*/280, /*last=*/1000},
            },

            .free_size = 728,
        },

        // TODO(apryakhin@): Consider enabling it back
        // Case 7: split range with duplicated tag
        /*RangeAllocTestCase{
            .alloc_ranges{{24, 1000}},
            .alloc_tags{1},
            .alloc_sizes{800},
            .alloc_aligns{24},

            .split_ops{{.old_tag = 1,
                        .good_allocs = 1,
                        .new_tags{7, 7},
                        .sizes{248, 248},
                        .aligns{16, 24},
                        .exp_offsets{32, 0}}},

            .exp_ranges{
                Range{0, 0},
                // free range after alignment computation
                Range{24, 32},
                Range{32, 280},
                Range{280, /1000},

            },
            .free_size = 728,
        },*/

        // Case 8: split range on different layouts and create free blocks in
        // front
        RangeAllocTestCase{
            .alloc_ranges{{24, 1000}},
            .alloc_tags{1},
            .alloc_sizes{800},
            .alloc_aligns{24},

            .split_ops{{.old_tag = 1,
                        .good_allocs = 2,
                        .new_tags{7, 8},
                        .sizes{248, 248},
                        .aligns{16, 24},
                        .exp_offsets{32, 288}}},

            .exp_ranges{
                Range{/*first=*/0, /*last=*/0},
                // free range after alignment computation
                Range{/*first=*/24, /*last=*/32},
                Range{/*first=*/32, /*last=*/280},
                // free range after alignment computation
                Range{/*first=*/280, /*last=*/288},
                Range{/*first=*/288, /*last=*/536},
                Range{/*first=*/536, /*last=*/1000},
            },
            .free_size = 480,
        },

        // Case 9: run multiple even splits
        RangeAllocTestCase{
            .alloc_ranges{{0, 256}},
            .alloc_tags{1, 2},
            .alloc_sizes{128, 128},
            .alloc_aligns{16, 16},

            .split_ops{{.old_tag = 1,
                        .good_allocs = 4,
                        .new_tags{41, 42, 43, 44},
                        .sizes{32, 32, 32, 32},
                        .aligns{16, 16, 16, 16},
                        .exp_offsets{0, 32, 64, 96}},

                       {.old_tag = 2,
                        .good_allocs = 4,
                        .new_tags{45, 46, 47, 48},
                        .sizes{32, 32, 32, 32},
                        .aligns{16, 16, 16, 16},
                        .exp_offsets{128, 160, 192, 224}}},

            .exp_ranges{
                Range{/*first=*/0, /*last=*/0},
                Range{/*first=*/0, /*last=*/32},
                Range{/*first=*/32, /*last=*/64},
                Range{/*first=*/64, /*last=*/96},
                Range{/*first=*/96, /*last=*/128},
                Range{/*first=*/128, /*last=*/128},
                Range{/*first=*/128, /*last=*/160},
                Range{/*first=*/160, /*last=*/192},
                Range{/*first=*/192, /*last=*/224},
                Range{/*first=*/224, /*last=*/256},
                Range{/*first=*/256, /*last=*/256},
            },
            .free_size = 0,
        },

        // Case 10: run multiple split that result in fragmentation
        RangeAllocTestCase{
            .alloc_ranges{{0, 256}},
            .alloc_tags{1, 2, 3, 4},
            .alloc_sizes{64, 64, 64, 64},
            .alloc_aligns{16, 16, 16, 16},

            .split_ops{
                {.old_tag = 1,
                 .good_allocs = 1,
                 .new_tags{41},
                 .sizes{60},
                 .aligns{1},
                 .exp_offsets{0}},
                {.old_tag = 2,
                 .good_allocs = 1,
                 .new_tags{42},
                 .sizes{60},
                 .aligns{1},
                 .exp_offsets{64}},
                {.old_tag = 3,
                 .good_allocs = 1,
                 .new_tags{43},
                 .sizes{60},
                 .aligns{1},
                 .exp_offsets{128}},
                {.old_tag = 4,
                 .good_allocs = 1,
                 .new_tags{44},
                 .sizes{60},
                 .aligns{1},
                 .exp_offsets{192}},
            },
            // TODO(apryakhin@): Check ranges
            .free_size = 16,
        },

        // Case 11: Reuse range and produce a free block due to
        // alignment followed by splitting the newly created range into
        // another set of ranges each of which also produce a free
        // block.
        RangeAllocTestCase{
            .alloc_ranges{{24, 1000}},
            .alloc_tags{1},
            .alloc_sizes{800},
            .alloc_aligns{24},

            .split_ops{

                {.old_tag = 1,
                 .good_allocs = 1,
                 .new_tags{7},
                 .sizes{496},
                 .aligns{16},
                 .exp_offsets{32}},

                {.old_tag = 7,
                 .good_allocs = 2,
                 .new_tags{8, 9},
                 .sizes{200, 200},
                 .aligns{12, 24},
                 .exp_offsets{36, 240}}},

            // TODO(apryakhin@): Check ranges
            .free_size = 576,
        }

        ));
