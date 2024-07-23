#include "realm/mem_impl.h"

#include <tuple>
#include <gtest/gtest.h>

using namespace Realm;

TEST(RangeAllocatorTestsWithParams, SplitRangeEmpty)
{
  BasicRangeAllocator<size_t, int> range_alloc;

  size_t offset = 0;
  EXPECT_FALSE(range_alloc.allocate(0, 512, 16, offset));

  std::vector<int> tags{2, 3};
  std::vector<size_t> sizes{512, 256};
  std::vector<size_t> alignment{16, 16};

  EXPECT_FALSE(range_alloc.split_range(1, tags, sizes, alignment));
}

TEST(RangeAllocatorTestsWithParams, SplitRangeIvalidID)
{
  BasicRangeAllocator<size_t, int> range_alloc;
  range_alloc.add_range(0, 1024);

  size_t offset = 0;
  EXPECT_TRUE(range_alloc.allocate(0, 512, 16, offset));

  std::vector<int> tags{2, 3};
  std::vector<size_t> sizes{512, 256};
  std::vector<size_t> alignment{16, 16};

  EXPECT_FALSE(range_alloc.split_range(1, tags, sizes, alignment));
}

TEST(RangeAllocatorTestsWithParams, SplitRangeInvalidSize)
{
  BasicRangeAllocator<size_t, int> range_alloc;
  range_alloc.add_range(0, 1024);

  size_t offset = 0;
  EXPECT_TRUE(range_alloc.allocate(0, 512, 16, offset));

  std::vector<int> tags{1, 2};
  std::vector<size_t> sizes{512, 256};
  std::vector<size_t> alignment{16, 16};

  EXPECT_FALSE(range_alloc.split_range(0, tags, sizes, alignment));
}

TEST(RangeAllocatorTestsWithParams, SplitRangeValid)
{
  BasicRangeAllocator<size_t, int> range_alloc;
  range_alloc.add_range(0, 1024);
  EXPECT_TRUE(range_alloc.can_allocate(0, 1024, 16));

  size_t offset = 0;
  EXPECT_TRUE(range_alloc.allocate(0, 512, 16, offset));

  {
    size_t start, size;
    EXPECT_TRUE(range_alloc.lookup(0, start, size));
    EXPECT_EQ(start, 0);
    EXPECT_EQ(size, 512);
  }

  std::vector<int> tags{1, 2};
  std::vector<size_t> sizes{256, 256};
  std::vector<size_t> alignment{16, 48};

  EXPECT_TRUE(range_alloc.split_range(0, tags, sizes, alignment));

  size_t exp_start = 0;
  for(size_t i = 0; i < tags.size(); i++) {
    size_t start, size;
    EXPECT_TRUE(range_alloc.lookup(tags[i], start, size));
    EXPECT_EQ(start, exp_start);
    EXPECT_EQ(size, sizes[i]);
    exp_start += size;
  }
}
