#include "realm/mem_impl.h"

#include <tuple>
#include <gtest/gtest.h>

using namespace Realm;

class RangeAllocatorTestsWithParams : public ::testing::Test {
protected:
  const unsigned int SENTINEL = BasicRangeAllocator<size_t, int>::SENTINEL;
};

TEST_F(RangeAllocatorTestsWithParams, DeallocateNonExistent)
{
  BasicRangeAllocator<size_t, int> range_alloc;
  range_alloc.deallocate(42, /*missiok_ok=*/true);
}

TEST_F(RangeAllocatorTestsWithParams, LookupEmptyAllocator)
{
  size_t start = 0, size = 0;
  BasicRangeAllocator<size_t, int> range_alloc;
  EXPECT_FALSE(range_alloc.lookup(0, start, size));
}

TEST_F(RangeAllocatorTestsWithParams, AllocateAndLookupInvalidRange)
{
  const int range_tag = 42;
  size_t offset = 0;
  size_t start = 0, size = 0;
  std::vector<int> tags{2, 3};
  std::vector<size_t> sizes{512, 256};
  std::vector<size_t> alignment{16, 16};
  BasicRangeAllocator<size_t, int> range_alloc;

  range_alloc.add_range(0, 1024);
  EXPECT_TRUE(range_alloc.allocate(range_tag, 512, 16, offset));
  EXPECT_FALSE(range_alloc.lookup(range_tag - 1, start, size));
}

TEST_F(RangeAllocatorTestsWithParams, AllocateAndLookupSingleRange)
{
  const int range_tag = 42;
  size_t offset = 0;
  size_t start = 0, size = 0;
  std::vector<int> tags{2, 3};
  std::vector<size_t> sizes{512, 256};
  std::vector<size_t> alignment{16, 16};
  BasicRangeAllocator<size_t, int> range_alloc;

  range_alloc.add_range(0, 1024);
  EXPECT_TRUE(range_alloc.allocate(range_tag, 512, 16, offset));
  EXPECT_TRUE(range_alloc.lookup(range_tag, start, size));

  EXPECT_EQ(start, 0);
  EXPECT_EQ(size, 512);
}

TEST_F(RangeAllocatorTestsWithParams, SplitRangeEmpty)
{
  size_t offset = 0;
  std::vector<int> tags{2, 3};
  std::vector<size_t> sizes{512, 256};
  std::vector<size_t> alignment{16, 16};
  std::vector<size_t> offsets{0, 0};
  BasicRangeAllocator<size_t, int> range_alloc;

  EXPECT_FALSE(range_alloc.allocate(0, 512, 16, offset));
  EXPECT_EQ(range_alloc.split_range(1, tags, sizes, alignment, offsets), 0);

  for(size_t i = 0; i < offsets.size(); i++) {
    EXPECT_EQ(offsets[0], 0);
  }
}

TEST_F(RangeAllocatorTestsWithParams, SplitRangeIvalidID)
{
  size_t offset = 0;
  std::vector<int> tags{2, 3};
  std::vector<size_t> sizes{512, 256};
  std::vector<size_t> alignment{16, 16};
  std::vector<size_t> offsets{0, 0};
  BasicRangeAllocator<size_t, int> range_alloc;

  range_alloc.add_range(0, 1024);
  EXPECT_TRUE(range_alloc.allocate(7, 512, 16, offset));

  EXPECT_EQ(range_alloc.split_range(8, tags, sizes, alignment, offsets), 0);

  for(size_t i = 0; i < offsets.size(); i++) {
    EXPECT_EQ(offsets[0], 0);
  }
}

TEST_F(RangeAllocatorTestsWithParams, SplitRangeInvalidSize)
{
  size_t offset = 0;
  std::vector<int> tags{1, 2};
  std::vector<size_t> sizes{512, 256};
  std::vector<size_t> alignment{16, 16};
  std::vector<size_t> offsets{0, 0};
  BasicRangeAllocator<size_t, int> range_alloc;

  range_alloc.add_range(0, 1024);
  EXPECT_TRUE(range_alloc.allocate(7, 512, 16, offset));

  EXPECT_EQ(range_alloc.split_range(7, tags, sizes, alignment, offsets), 1);

  for(size_t i = 0; i < offsets.size(); i++) {
    EXPECT_EQ(offsets[0], 0);
  }
}

// TODO: alignment test
// requested range with alignment should exceed the existing range
TEST_F(RangeAllocatorTestsWithParams, ReuseFailureDueToAlignment)
{
  const size_t old_range_size = 200;
  const int old_range_alignment = 1;
  const int old_range_tag = 42;
  const int new_range_tag = 43;
  size_t offset = 0;
  std::vector<int> tags{new_range_tag};
  std::vector<size_t> sizes{100};
  std::vector<size_t> alignment{16};
  std::vector<size_t> offsets{0};
  BasicRangeAllocator<size_t, int> range_alloc;

  range_alloc.add_range(0, old_range_size);
  EXPECT_TRUE(range_alloc.allocate(old_range_tag, 100, old_range_alignment, offset));
  EXPECT_TRUE(range_alloc.allocate(old_range_tag + 1, 100, old_range_alignment, offset));

  EXPECT_EQ(range_alloc.split_range(old_range_tag + 1, tags, sizes, alignment, offsets),
            0);
}

TEST_F(RangeAllocatorTestsWithParams, ReuseZeroRange)
{
  const int old_range_tag = 42;
  const int new_range_tag = 43;
  size_t offset = 0;
  size_t old_start = 0, old_size = 0;
  size_t new_start = 0, new_size = 0;
  std::vector<int> tags{new_range_tag};
  std::vector<size_t> sizes{0};
  std::vector<size_t> alignment{16};
  std::vector<size_t> offsets{0};
  BasicRangeAllocator<size_t, int> range_alloc;

  range_alloc.add_range(0, 2);
  EXPECT_TRUE(range_alloc.allocate(old_range_tag, 0, 1, offset));
  EXPECT_TRUE(range_alloc.allocate(old_range_tag + 1, 0, 1, offset));
  const size_t num_ranges = range_alloc.ranges.size();
  EXPECT_EQ(range_alloc.split_range(old_range_tag, tags, sizes, alignment, offsets),
            tags.size());
  EXPECT_FALSE(range_alloc.lookup(old_range_tag, old_start, old_size));
  EXPECT_TRUE(range_alloc.lookup(new_range_tag, new_start, new_size));

  EXPECT_EQ(range_alloc.ranges.size(), num_ranges);
  EXPECT_EQ(offsets.size(), 1);
  EXPECT_EQ(offsets[0], 0);
  EXPECT_EQ(old_start, 0);
  EXPECT_EQ(old_size, 0);
  EXPECT_EQ(new_start, 0);
  EXPECT_EQ(new_size, 0);
}

TEST_F(RangeAllocatorTestsWithParams, ReuseRange)
{
  const int old_range_tag = 42;
  const int new_range_tag = 43;
  size_t offset = 0;
  size_t old_start = 0, old_size = 0;
  size_t new_start = 0, new_size = 0;
  std::vector<int> tags{new_range_tag};
  std::vector<size_t> sizes{512};
  std::vector<size_t> alignment{16};
  std::vector<size_t> offsets{0};
  BasicRangeAllocator<size_t, int> range_alloc;

  range_alloc.add_range(0, 512);
  EXPECT_TRUE(range_alloc.allocate(old_range_tag, 512, 16, offset));
  EXPECT_TRUE(range_alloc.lookup(old_range_tag, old_start, old_size));
  EXPECT_EQ(range_alloc.split_range(old_range_tag, tags, sizes, alignment, offsets),
            tags.size());
  EXPECT_FALSE(range_alloc.lookup(old_range_tag, old_start, old_size));
  EXPECT_TRUE(range_alloc.lookup(new_range_tag, new_start, new_size));

  EXPECT_EQ(new_start, old_start);
  EXPECT_EQ(new_size, old_size);
  EXPECT_EQ(range_alloc.ranges.size(), 2);
  EXPECT_EQ(range_alloc.ranges[0].last, 0);
  EXPECT_EQ(range_alloc.ranges[0].prev, 1);
  EXPECT_EQ(range_alloc.ranges[0].next, 1);
  EXPECT_EQ(range_alloc.ranges[0].prev_free, SENTINEL);
  EXPECT_EQ(range_alloc.ranges[0].next_free, SENTINEL);

  EXPECT_EQ(range_alloc.ranges[1].first, offsets[0]);
  EXPECT_EQ(range_alloc.ranges[1].first, new_start);
  EXPECT_EQ(range_alloc.ranges[1].last, new_start + new_size);
  EXPECT_EQ(range_alloc.ranges[1].prev, 0);
  EXPECT_EQ(range_alloc.ranges[1].next, 0);
  EXPECT_EQ(range_alloc.ranges[1].prev_free, 1);
  EXPECT_EQ(range_alloc.ranges[1].next_free, 1);
}

// TODO: Test when splitting on smaller size

TEST_F(RangeAllocatorTestsWithParams, SplitRange)
{
  const int old_range_tag = 42;
  size_t offset = 0;
  std::vector<int> tags{7, 14};
  std::vector<size_t> sizes{256, 256};
  std::vector<size_t> alignment{16, 16};
  std::vector<size_t> offsets{0, 0};
  std::vector<size_t> new_starts(2);
  std::vector<size_t> new_sizes(2);
  std::vector<size_t> new_offsets(2);
  BasicRangeAllocator<size_t, int> range_alloc;

  range_alloc.add_range(0, 768);
  EXPECT_TRUE(range_alloc.allocate(old_range_tag, 512, 16, offset));
  EXPECT_EQ(range_alloc.split_range(old_range_tag, tags, sizes, alignment, offsets),
            tags.size());
  for(size_t i = 0; i < tags.size(); i++) {
    EXPECT_TRUE(range_alloc.lookup(tags[i], new_starts[i], new_sizes[i]));
  }

  EXPECT_EQ(range_alloc.ranges.size(), 4);

  {
    // check .first
    EXPECT_EQ(range_alloc.ranges[0].last, 0);
    EXPECT_EQ(range_alloc.ranges[0].prev, 2);
    EXPECT_EQ(range_alloc.ranges[0].next, 1);
    EXPECT_EQ(range_alloc.ranges[0].prev_free, 2);
    EXPECT_EQ(range_alloc.ranges[0].next_free, 2);
  }

  {
    EXPECT_EQ(range_alloc.ranges[1].first, offsets[0]);
    EXPECT_EQ(range_alloc.ranges[1].first, new_starts[0]);
    EXPECT_EQ(range_alloc.ranges[1].last, new_starts[0] + new_sizes[0]);
    EXPECT_EQ(range_alloc.ranges[1].prev, 0);
    EXPECT_EQ(range_alloc.ranges[1].next, 3);
    EXPECT_EQ(range_alloc.ranges[1].prev_free, 1);
    EXPECT_EQ(range_alloc.ranges[1].next_free, 1);
  }

  {
    EXPECT_EQ(range_alloc.ranges[2].first, 512);
    EXPECT_EQ(range_alloc.ranges[2].last, 768);
    EXPECT_EQ(range_alloc.ranges[2].prev, 3);
    EXPECT_EQ(range_alloc.ranges[2].next, 0);
    EXPECT_EQ(range_alloc.ranges[2].prev_free, SENTINEL);
    EXPECT_EQ(range_alloc.ranges[2].next_free, SENTINEL);
  }

  {
    EXPECT_EQ(range_alloc.ranges[3].first, offsets[1]);
    EXPECT_EQ(range_alloc.ranges[3].first, new_starts[1]);
    EXPECT_EQ(range_alloc.ranges[3].last, new_starts[1] + new_sizes[1]);
    EXPECT_EQ(range_alloc.ranges[3].prev, 1);
    EXPECT_EQ(range_alloc.ranges[3].next, 2);
    EXPECT_EQ(range_alloc.ranges[3].prev_free, 3);
    EXPECT_EQ(range_alloc.ranges[3].next_free, 3);
  }
}

TEST_F(RangeAllocatorTestsWithParams, SplitRangeAndProduceFreeRange)
{
  const int old_range_tag = 42;
  size_t offset = 0;
  std::vector<int> tags{7, 14};
  std::vector<size_t> sizes{256, 256};
  std::vector<size_t> alignment{16, 16};
  std::vector<size_t> offsets{0, 0};
  std::vector<size_t> new_starts(2);
  std::vector<size_t> new_sizes(2);
  std::vector<size_t> new_offsets(2);
  BasicRangeAllocator<size_t, int> range_alloc;

  range_alloc.add_range(0, 768);
  EXPECT_TRUE(range_alloc.allocate(old_range_tag, 768, 16, offset));
  EXPECT_EQ(range_alloc.split_range(old_range_tag, tags, sizes, alignment, offsets),
            tags.size());
  for(size_t i = 0; i < tags.size(); i++) {
    EXPECT_TRUE(range_alloc.lookup(tags[i], new_starts[i], new_sizes[i]));
  }

  EXPECT_EQ(range_alloc.ranges.size(), 4);

  {
    // check .first
    EXPECT_EQ(range_alloc.ranges[0].last, 0);
    EXPECT_EQ(range_alloc.ranges[0].prev, 3);
    EXPECT_EQ(range_alloc.ranges[0].next, 1);
    EXPECT_EQ(range_alloc.ranges[0].prev_free, 3);
    EXPECT_EQ(range_alloc.ranges[0].next_free, 3);
  }

  {
    EXPECT_EQ(range_alloc.ranges[1].first, offsets[0]);
    EXPECT_EQ(range_alloc.ranges[1].first, new_starts[0]);
    EXPECT_EQ(range_alloc.ranges[1].last, new_starts[0] + new_sizes[0]);
    EXPECT_EQ(range_alloc.ranges[1].prev, 0);
    EXPECT_EQ(range_alloc.ranges[1].next, 2);
    EXPECT_EQ(range_alloc.ranges[1].prev_free, 1);
    EXPECT_EQ(range_alloc.ranges[1].next_free, 1);
  }

  {
    EXPECT_EQ(range_alloc.ranges[2].first, offsets[1]);
    EXPECT_EQ(range_alloc.ranges[2].first, new_starts[1]);
    EXPECT_EQ(range_alloc.ranges[2].last, new_starts[1] + new_sizes[1]);
    EXPECT_EQ(range_alloc.ranges[2].prev, 1);
    EXPECT_EQ(range_alloc.ranges[2].next, 3);
    EXPECT_EQ(range_alloc.ranges[2].prev_free, 2);
    EXPECT_EQ(range_alloc.ranges[2].next_free, 2);
  }

  {
    EXPECT_EQ(range_alloc.ranges[3].first, 512);
    EXPECT_EQ(range_alloc.ranges[3].last, 768);
    EXPECT_EQ(range_alloc.ranges[3].prev, 2);
    EXPECT_EQ(range_alloc.ranges[3].next, 0);
    EXPECT_EQ(range_alloc.ranges[3].prev_free, SENTINEL);
    EXPECT_EQ(range_alloc.ranges[3].next_free, SENTINEL);
  }
}

// TODO(apryakhin@): Add test case to show the fragmentation after
// multiple split_range calls.
TEST_F(RangeAllocatorTestsWithParams, ReuseRangeAndProduceFreeRange)
{
  int old_range_tag = 1;
  size_t offset = 0;
  std::vector<int> tags{101};
  std::vector<size_t> sizes{16};
  std::vector<size_t> alignment{1};
  std::vector<size_t> offsets{0};
  BasicRangeAllocator<size_t, int> range_alloc;
  range_alloc.add_range(0, 512);

  for(int i = 0; i < 10; i++) {
    EXPECT_TRUE(range_alloc.allocate(old_range_tag, sizes[0] * 2, 1, offset));
    EXPECT_EQ(range_alloc.split_range(old_range_tag, tags, sizes, alignment, offsets),
              tags.size());
    old_range_tag++;
    tags[0]++;
  }

  EXPECT_EQ(range_alloc.ranges.size(), 22);
}

TEST_F(RangeAllocatorTestsWithParams, TestExplicitRangesWithCycle)
{
  BasicRangeAllocator<size_t, int> alloc;

  {
    alloc.ranges[SENTINEL].prev_free = 3;
    alloc.ranges[SENTINEL].next_free = 1;
  }
  {
    BasicRangeAllocator<size_t, int>::Range range;
    range.prev_free = SENTINEL;
    range.next_free = 2;
    alloc.ranges.push_back(range);
  }
  {
    BasicRangeAllocator<size_t, int>::Range range;
    range.prev_free = 1;
    range.next_free = 3;
    alloc.ranges.push_back(range);
  }
  {
    BasicRangeAllocator<size_t, int>::Range range;
    // Link up with range 1
    range.prev_free = 2;
    range.next_free = 1;
    alloc.ranges.push_back(range);
  }

  EXPECT_TRUE(alloc.free_list_has_cycle());
}

TEST_F(RangeAllocatorTestsWithParams, TestExplicitRangesNoCycle)
{
  BasicRangeAllocator<size_t, int> alloc;

  {
    alloc.ranges[SENTINEL].prev_free = 3;
    alloc.ranges[SENTINEL].next_free = 1;
  }
  {
    BasicRangeAllocator<size_t, int>::Range range;
    range.prev_free = SENTINEL;
    range.next_free = 2;
    alloc.ranges.push_back(range);
  }
  {
    BasicRangeAllocator<size_t, int>::Range range;
    range.prev_free = 1;
    range.next_free = 3;
    alloc.ranges.push_back(range);
  }
  {
    BasicRangeAllocator<size_t, int>::Range range;
    range.prev_free = 2;
    range.next_free = SENTINEL;
    alloc.ranges.push_back(range);
  }

  EXPECT_FALSE(alloc.free_list_has_cycle());
}

TEST_F(RangeAllocatorTestsWithParams, TestExplicitRangesOverlapping)
{
  BasicRangeAllocator<size_t, int> alloc;

  {
    alloc.ranges[SENTINEL].prev = 3;
    alloc.ranges[SENTINEL].next = 1;
  }
  {
    BasicRangeAllocator<size_t, int>::Range range;
    range.prev = SENTINEL;
    range.next = 2;
    range.first = 0;
    range.last = 16;
    alloc.ranges.push_back(range);
    alloc.allocated[1] = 1;
  }
  {
    BasicRangeAllocator<size_t, int>::Range range;
    range.prev = 1;
    range.next = 3;
    range.first = 16;
    range.last = 32;
    alloc.ranges.push_back(range);
    alloc.allocated[2] = 2;
  }
  {
    BasicRangeAllocator<size_t, int>::Range range;
    range.prev = 2;
    range.next = SENTINEL;
    range.first = 16;
    range.last = 64;
    alloc.ranges.push_back(range);
    alloc.allocated[3] = 3;
  }

  EXPECT_TRUE(alloc.has_invalid_ranges());
}

TEST_F(RangeAllocatorTestsWithParams, TestExplicitRangesInvalidSize)
{
  BasicRangeAllocator<size_t, int> alloc;

  {
    alloc.ranges[SENTINEL].prev = 3;
    alloc.ranges[SENTINEL].next = 1;
  }
  {
    BasicRangeAllocator<size_t, int>::Range range;
    range.prev = SENTINEL;
    range.next = 2;
    range.first = 0;
    range.last = 16;
    alloc.ranges.push_back(range);
    alloc.allocated[1] = 1;
  }
  {
    BasicRangeAllocator<size_t, int>::Range range;
    range.prev = 1;
    range.next = 3;
    range.first = 16;
    range.last = 15;
    alloc.ranges.push_back(range);
    alloc.allocated[2] = 2;
  }
  {
    BasicRangeAllocator<size_t, int>::Range range;
    range.prev = 2;
    range.next = SENTINEL;
    range.first = 32;
    range.last = 64;
    alloc.ranges.push_back(range);
    alloc.allocated[3] = 3;
  }

  EXPECT_TRUE(alloc.has_invalid_ranges());
}

TEST_F(RangeAllocatorTestsWithParams, TestExplicitRangesValid)
{
  BasicRangeAllocator<size_t, int> alloc;

  {
    alloc.ranges[SENTINEL].prev = 3;
    alloc.ranges[SENTINEL].next = 1;
  }
  {
    BasicRangeAllocator<size_t, int>::Range range;
    range.prev = SENTINEL;
    range.next = 2;
    range.first = 0;
    range.last = 16;
    alloc.ranges.push_back(range);
    alloc.allocated[1] = 1;
  }
  {
    BasicRangeAllocator<size_t, int>::Range range;
    range.prev = 1;
    range.next = 3;
    range.first = 16;
    range.last = 20;
    alloc.ranges.push_back(range);
    alloc.allocated[2] = 2;
  }
  {
    BasicRangeAllocator<size_t, int>::Range range;
    range.prev = 2;
    range.next = SENTINEL;
    range.first = 32;
    range.last = 64;
    alloc.ranges.push_back(range);
    alloc.allocated[3] = 3;
  }

  EXPECT_FALSE(alloc.has_invalid_ranges());
}

TEST_F(RangeAllocatorTestsWithParams, TestExplicitRangesSharedTags)
{
  BasicRangeAllocator<size_t, int> alloc;

  {
    alloc.ranges[SENTINEL].prev = 3;
    alloc.ranges[SENTINEL].next = 1;
  }
  {
    BasicRangeAllocator<size_t, int>::Range range;
    range.prev = SENTINEL;
    range.next = 2;
    range.first = 0;
    range.last = 16;
    alloc.ranges.push_back(range);
    alloc.allocated[1] = 1;
  }
  {
    BasicRangeAllocator<size_t, int>::Range range;
    range.prev = 1;
    range.next = 3;
    range.first = 16;
    range.last = 20;
    alloc.ranges.push_back(range);
    alloc.allocated[2] = 2;
  }
  {
    BasicRangeAllocator<size_t, int>::Range range;
    range.prev = 2;
    range.next = SENTINEL;
    range.first = 32;
    range.last = 64;
    alloc.ranges.push_back(range);
    alloc.allocated[3] = 2;
  }

  EXPECT_TRUE(alloc.has_invalid_ranges());
}
