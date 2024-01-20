#include "realm/transfer/lowlevel_dma.h"
#include <tuple>
#include <gtest/gtest.h>

using namespace Realm;

TEST(PathCacheTest, ZeroEntries)
{
  const size_t num_entries = 0;
  const size_t num_bytes = 10;

  PathLRU cache(num_entries);
  PathLRU::LRUKey key(0, 0, num_bytes, {}, {});
  PathLRU::PathLRUIterator it = cache.find(key);
  EXPECT_TRUE(it == cache.end());
  // seg-fault on zero size cache miss
  // cache.miss(key, MemPathInfo());
}

TEST(PathCacheTest, DoubleMissCheckValueUpdate)
{
  const size_t num_entries = 1;
  const size_t num_bytes = 10;

  PathLRU cache(num_entries);
  PathLRU::LRUKey key(0, 0, num_bytes, {}, {});

  int num_paths = 1;

  {
    MemPathInfo info;
    info.path.push_back(Memory());
    cache.miss(key, info);
    PathLRU::PathLRUIterator it = cache.find(key);
    EXPECT_EQ((*it).second.path.size(), num_paths);
  }

  num_paths++;

  {
    MemPathInfo info;
    info.path.push_back(Memory());
    info.path.push_back(Memory());
    cache.miss(key, info);
    PathLRU::PathLRUIterator it = cache.find(key);
    EXPECT_EQ((*it).second.path.size(), num_paths);
  }
}

TEST(PathCacheTest, EvictLastRecentlyUsedEntry)
{
  const size_t num_entries = 7;
  const size_t num_bytes = 10;
  const size_t evict_idx = 4;

  PathLRU cache(num_entries);
  for(size_t i = 0; i < num_entries; i++) {
    PathLRU::LRUKey key(0, 0, num_bytes + i, {}, {});
    PathLRU::PathLRUIterator it = cache.find(key);
    EXPECT_TRUE(it == cache.end());
    cache.miss(key, MemPathInfo());
  }

  // hit all the entries except one
  for(size_t i = 0; i < num_entries; i++) {
    if(i == evict_idx)
      continue;
    PathLRU::LRUKey key(0, 0, num_bytes + i, {}, {});
    PathLRU::PathLRUIterator it = cache.find(key);
    EXPECT_TRUE(it != cache.end());
    cache.hit(it);
  }

  // make sure this entry is still in the cache
  {
    PathLRU::LRUKey key(0, 0, num_bytes + evict_idx, {}, {});
    PathLRU::PathLRUIterator it = cache.find(key);
    EXPECT_TRUE(it != cache.end());
  }

  // add another entry to grow beyond num_entries
  {
    PathLRU::LRUKey key(0, 0, num_bytes + num_entries, {}, {});
    cache.miss(key, MemPathInfo());
  }

  // entry must have bee evicted
  {
    PathLRU::LRUKey key(0, 0, num_bytes + evict_idx, {}, {});
    PathLRU::PathLRUIterator it = cache.find(key);
    EXPECT_TRUE(it == cache.end());
  }
}
