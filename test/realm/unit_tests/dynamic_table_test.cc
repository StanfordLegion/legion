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
};

typedef DynamicTableAllocator<Dummy, 0, 4> DummyTableAllocator;

TEST(DynamicTableTest, Create1DEntry) {}
