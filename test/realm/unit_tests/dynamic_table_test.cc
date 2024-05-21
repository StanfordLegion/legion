#include <cstddef>
#include "realm/id.h"
#include "realm/mutex.h"

#include "realm/event_impl.h"
#include "realm/rsrv_impl.h"
#include "realm/deppart/sparsity_impl.h"
#include "realm/subgraph_impl.h"
#include "realm/dynamic_table.h"
#include "realm/dynamic_table_allocator.h"

#include <gtest/gtest.h>

using namespace Realm;

struct Dummy {
  void init(ID _me, unsigned _init_owner) { me = me; }
  ID me;
  Dummy *next_free = nullptr;
};

TEST(DynamicTableTest, CheckEmptyDynamicTable)
{
  const int leaf_bits = 4;
  DynamicTable<DynamicTableAllocator<Dummy, 1, leaf_bits>> dtable;
  int id = 0;

  EXPECT_FALSE(dtable.has_entry(id));
  EXPECT_EQ(dtable.max_entries(), 0);
}

TEST(DynamicTableTest, LookupSingleEntry)
{
  const int leaf_bits = 4;
  DynamicTable<DynamicTableAllocator<Dummy, 1, leaf_bits>> dtable;
  int id = 0;

  dtable.lookup_entry(id, 0);

  EXPECT_TRUE(dtable.has_entry(id));
  EXPECT_EQ(dtable.max_entries(), 16);
}

TEST(DynamicTableTest, LookupMultipleEntries)
{
  const int leaf_bits = 4;
  DynamicTable<DynamicTableAllocator<Dummy, 1, leaf_bits>> dtable;
  std::vector<int> ids { 0, 15, 16, 33 };
  int id0 = 0, id1 = 15, id2 = 16, id3 = 33;

  std::vector<Dummy *> entries(ids.size());
  for(int i = 0; i < ids.size(); i++) {
    entries[i] = dtable.lookup_entry(ids[i], 0);
  }

  EXPECT_EQ(dtable.max_entries(), 64);
  for(int i = 0; i < ids.size(); i++) {
    EXPECT_TRUE(dtable.has_entry(ids[i]));
    EXPECT_NE(entries[i], nullptr);
  }
}

TEST(DynamicTableTest, LookupTheLimit)
{
  const int leaf_bits = 4;
  const int max_levels = 7;
  DynamicTable<DynamicTableAllocator<Dummy, 1, leaf_bits>> dtable;
  int index = 0;

  std::vector<Dummy *> entries(((1 << leaf_bits) << max_levels));
  for(int i = 0; i < ((1 << leaf_bits) << max_levels); i++) {
    entries[i] = dtable.lookup_entry(index++, 0);
  }

  EXPECT_EQ(dtable.max_entries(), index);
  for(int i = 0; i < entries.size(); i++) {
    EXPECT_NE(entries[i], nullptr);
  }
}

TEST(DynamicTableTest, FreeListSingleAlloc)
{
  const int leaf_bits = 4;
  DynamicTable<DynamicTableAllocator<Dummy, 1, leaf_bits>> dtable;
  DynamicTableAllocator<Dummy, 1, leaf_bits>::FreeList free_list(dtable, 0);

  Dummy *entry = free_list.alloc_entry();

  EXPECT_NE(entry, nullptr);
  EXPECT_EQ(entry->next_free, nullptr);
  EXPECT_TRUE(dtable.has_entry(entry->me.id));
}

TEST(DynamicTableTest, FreeListSingleAllocAndFree)
{
  const int leaf_bits = 4;
  DynamicTable<DynamicTableAllocator<Dummy, 1, leaf_bits>> dtable;
  DynamicTableAllocator<Dummy, 1, leaf_bits>::FreeList free_list(dtable, 0);

  Dummy *entry = free_list.alloc_entry();
  free_list.free_entry(entry);

  EXPECT_NE(entry, nullptr);
  EXPECT_NE(entry->next_free, nullptr);
  EXPECT_TRUE(dtable.has_entry(entry->me.id));
}

TEST(DynamicTableTest, FreeListUpMaxAlloc)
{
  const int leaf_bits = 4;
  DynamicTable<DynamicTableAllocator<Dummy, 1, leaf_bits>> dtable;
  DynamicTableAllocator<Dummy, 1, leaf_bits>::FreeList free_list(dtable, 0);

  std::vector<Dummy *> entries(255);
  for(int i = 0; i < 255; i++) {
    entries[i] = free_list.alloc_entry();
  }

  for(int i = 0; i < 255; i++) {
    EXPECT_NE(entries[i], nullptr);
    EXPECT_TRUE(dtable.has_entry(entries[i]->me.id));
  }
}

TEST(DynamicTableTest, FreeListOverMaxAlloc)
{
  const int leaf_bits = 4;
  DynamicTable<DynamicTableAllocator<Dummy, 1, leaf_bits>> dtable;
  DynamicTableAllocator<Dummy, 1, leaf_bits>::FreeList free_list(dtable, 0);

  std::vector<Dummy *> entries(256);
  for(int i = 0; i < 256; i++) {
    entries[i] = free_list.alloc_entry();
    free_list.free_entry(entries[i]);
  }

  for(int i = 0; i < 256; i++) {
    EXPECT_NE(entries[i], nullptr);
    EXPECT_TRUE(dtable.has_entry(entries[i]->me.id));
  }
}
