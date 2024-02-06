#include "realm.h"
#include "realm/sparsity.h"
#include <tuple>
#include <gtest/gtest.h>

using namespace Realm;

class SparsityMapTest : public ::testing::TestWithParam<int> {
protected:
  static void SetUpTestSuite()
  {
    runtime_ = new Runtime();
    runtime_->init(0, 0);
  }

  static void TearDownTestSuite()
  {
    runtime_->shutdown();
    delete runtime_;
  }

  void SetUp() override
  {
    rects.push_back(Rect<1>(Point<1>(0), Point<1>(4)));
    rects.push_back(Rect<1>(Point<1>(8), Point<1>(12)));
    rects.push_back(Rect<1>(Point<1>(16), Point<1>(18)));
  }

  std::vector<Rect<1>> rects;
  static Runtime *runtime_;
};

Runtime *SparsityMapTest::runtime_ = nullptr;

TEST_F(SparsityMapTest, Destroy)
{
  SparsityMap<1> sparsity_map = SparsityMap<1>::construct(rects, true, true);
  sparsity_map.destroy();
  auto *impl = sparsity_map.impl();
  EXPECT_NE(impl, nullptr);
  EXPECT_FALSE(impl->is_valid());
}

TEST_F(SparsityMapTest, DestroyWithEvent)
{
  SparsityMap<1> sparsity_map = SparsityMap<1>::construct(rects, true, true);
  UserEvent event = UserEvent::create_user_event();
  sparsity_map.destroy(event);
  auto *impl = sparsity_map.impl();
  EXPECT_NE(impl, nullptr);
  EXPECT_TRUE(impl->is_valid());
  event.trigger();
  event.wait();
  EXPECT_FALSE(impl->is_valid());
}

TEST_F(SparsityMapTest, DoubleDestroy)
{
  SparsityMap<1> sparsity_map = SparsityMap<1>::construct(rects, true, true);
  sparsity_map.destroy();
  sparsity_map.destroy();
  auto *impl = sparsity_map.impl();
  EXPECT_NE(impl, nullptr);
  EXPECT_FALSE(impl->is_valid());
}

TEST_F(SparsityMapTest, RemoveReference)
{
  SparsityMap<1> sparsity_map = SparsityMap<1>::construct(rects, true, true);
  sparsity_map.remove_references();
  auto *impl = sparsity_map.impl();
  EXPECT_NE(impl, nullptr);
  EXPECT_FALSE(impl->is_valid());
}

TEST_F(SparsityMapTest, RemoveZeroReferences)
{
  SparsityMap<1> sparsity_map = SparsityMap<1>::construct(rects, true, true);
  sparsity_map.remove_references(0);
  auto *impl = sparsity_map.impl();
  EXPECT_NE(impl, nullptr);
  EXPECT_FALSE(impl->is_valid());
}

TEST_F(SparsityMapTest, RemoveTwoReferences)
{
  SparsityMap<1> sparsity_map = SparsityMap<1>::construct(rects, true, true);
  sparsity_map.remove_references(2);
  auto *impl = sparsity_map.impl();
  EXPECT_NE(impl, nullptr);
  EXPECT_FALSE(impl->is_valid());
}

TEST_F(SparsityMapTest, AddRemoveReference)
{
  SparsityMap<1> sparsity_map = SparsityMap<1>::construct(rects, true, true);
  sparsity_map.add_references();
  sparsity_map.remove_references();
  auto *impl = sparsity_map.impl();
  EXPECT_NE(impl, nullptr);
  EXPECT_FALSE(impl->is_valid());
}

TEST_F(SparsityMapTest, AddRemoveZeroReference)
{
  SparsityMap<1> sparsity_map = SparsityMap<1>::construct(rects, true, true);
  sparsity_map.add_references();
  sparsity_map.remove_references(0);
  auto *impl = sparsity_map.impl();
  EXPECT_NE(impl, nullptr);
  EXPECT_TRUE(impl->is_valid());
}

TEST_F(SparsityMapTest, DoubleAddRemoveReference)
{
  SparsityMap<1> sparsity_map = SparsityMap<1>::construct(rects, true, true);
  sparsity_map.add_references();
  sparsity_map.add_references();
  sparsity_map.remove_references();
  auto *impl = sparsity_map.impl();
  EXPECT_NE(impl, nullptr);
  EXPECT_TRUE(impl->is_valid());
}

TEST_F(SparsityMapTest, DoubleAddRemoveTwoReference)
{
  SparsityMap<1> sparsity_map = SparsityMap<1>::construct(rects, true, true);
  sparsity_map.add_references(2);
  sparsity_map.remove_references(2);
  auto *impl = sparsity_map.impl();
  EXPECT_NE(impl, nullptr);
  EXPECT_FALSE(impl->is_valid());
}

TEST_F(SparsityMapTest, DestroyWithReference)
{
  SparsityMap<1> sparsity_map = SparsityMap<1>::construct(rects, true, true);
  sparsity_map.add_references();
  sparsity_map.destroy();
  auto *impl = sparsity_map.impl();
  EXPECT_NE(impl, nullptr);
  EXPECT_FALSE(impl->is_valid());
}

TEST_F(SparsityMapTest, DestroyWithDoubleReference)
{
  SparsityMap<1> sparsity_map = SparsityMap<1>::construct(rects, true, true);
  sparsity_map.add_references();
  sparsity_map.add_references();
  sparsity_map.destroy();
  auto *impl = sparsity_map.impl();
  EXPECT_NE(impl, nullptr);
  EXPECT_TRUE(impl->is_valid());
}
