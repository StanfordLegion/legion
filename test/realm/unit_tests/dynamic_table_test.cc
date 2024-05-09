#include <cstddef>
#include "realm/id.h"
#include "realm/mutex.h"

#include "realm/event_impl.h"
#include "realm/rsrv_impl.h"
#include "realm/deppart/sparsity_impl.h"
#include "realm/subgraph_impl.h"
#include "realm/dynamic_table.h"
#include "realm/dynamic_table_allocator.h"

#include <tuple>
#include <gtest/gtest.h>

using namespace Realm;

struct Dummy {
  void init(ID _me, unsigned _init_owner) {}
  Dummy *next_free;
  atomic<DynamicTemplates::TagType> type_tag;
};

typedef DynamicTableAllocator<Dummy, 1, 4> DummyTableAllocator;

TEST(DynamicTableTest, EmptyTable)
{
  DynamicTable<DummyTableAllocator> dtable;
  int id = 0;
  EXPECT_FALSE(dtable.has_entry(id));
  EXPECT_EQ(dtable.max_entries(), 0);
  dtable.lookup_entry(id, 0);
  EXPECT_TRUE(dtable.has_entry(id));
}

TEST(DynamicTableTest, LookupSingleEntry)
{
  DynamicTable<DummyTableAllocator> dtable;
  int id = 0;
  dtable.lookup_entry(id, 0);
  EXPECT_TRUE(dtable.has_entry(id));
  EXPECT_EQ(dtable.max_entries(), 16);
}

TEST(DynamicTableTest, LookupMultipleEntries)
{
  DynamicTable<DynamicTableAllocator<Dummy, 1, 4>> dtable;

  int id0 = 0;
  dtable.lookup_entry(id0, 0);
  EXPECT_TRUE(dtable.has_entry(id0));

  EXPECT_EQ(dtable.max_entries(), 16);

  int id1 = 15;
  EXPECT_TRUE(dtable.has_entry(id1));

  int id2 = 16;
  EXPECT_FALSE(dtable.has_entry(id2));
  // create new leaf node
  dtable.lookup_entry(id2, 0);
  EXPECT_TRUE(dtable.has_entry(id2));

  EXPECT_EQ(dtable.max_entries(), 32);

  int id3 = 33;
  EXPECT_FALSE(dtable.has_entry(id3));
  // create new leaf node
  dtable.lookup_entry(id3, 0);
  EXPECT_TRUE(dtable.has_entry(id3));

  EXPECT_EQ(dtable.max_entries(), 64);
}

TEST(DynamicTableTest, LookupTheLimit)
{
  DynamicTable<DynamicTableAllocator<Dummy, 1, 4>> dtable;
  int index = 0; // 2048
  int leaf_bits = (1 << 4);
  for(int i = 0; i < (1 << 7); i++) { // max levels
    for(int j = 0; j < (1 << 4); j++) {
      dtable.lookup_entry(index, 0);
      EXPECT_TRUE(dtable.has_entry(index++));
    }
  }
  EXPECT_EQ(dtable.max_entries(), index);
}
