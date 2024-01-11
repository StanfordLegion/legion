#include "realm.h"
#include "realm/sparsity.h"
#include <tuple>
#include <gtest/gtest.h>

using namespace Realm;

class TransferIteratorTest : public ::testing::TestWithParam<int> {
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

Runtime *TransferIteratorTest::runtime_ = nullptr;

TEST_F(TransferIteratorTest, Destroy)
{
  SparsityMap<1> sparsity_map = SparsityMap<1>::construct(rects, true, true);
  sparsity_map.destroy();
  auto *impl = sparsity_map.impl();
  EXPECT_NE(impl, nullptr);
  EXPECT_FALSE(impl->is_valid());
}

TEST_F(TransferIteratorTest, DestroyWithEvent)
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

TEST_F(TransferIteratorTest, DoubleDestroy)
{
  SparsityMap<1> sparsity_map = SparsityMap<1>::construct(rects, true, true);
  sparsity_map.destroy();
  sparsity_map.destroy();
  auto *impl = sparsity_map.impl();
  EXPECT_NE(impl, nullptr);
  EXPECT_FALSE(impl->is_valid());
}

TEST_F(TransferIteratorTest, RemoveReference)
{
  SparsityMap<1> sparsity_map = SparsityMap<1>::construct(rects, true, true);
  sparsity_map.remove_references();
  auto *impl = sparsity_map.impl();
  EXPECT_NE(impl, nullptr);
  EXPECT_FALSE(impl->is_valid());
}

TEST_F(TransferIteratorTest, RemoveZeroReferences)
{
  SparsityMap<1> sparsity_map = SparsityMap<1>::construct(rects, true, true);
  sparsity_map.remove_references(0);
  auto *impl = sparsity_map.impl();
  EXPECT_NE(impl, nullptr);
  EXPECT_FALSE(impl->is_valid());
}

TEST_F(TransferIteratorTest, RemoveTwoReferences)
{
  SparsityMap<1> sparsity_map = SparsityMap<1>::construct(rects, true, true);
  sparsity_map.remove_references(2);
  auto *impl = sparsity_map.impl();
  EXPECT_NE(impl, nullptr);
  EXPECT_FALSE(impl->is_valid());
}

TEST_F(TransferIteratorTest, AddRemoveReference)
{
  SparsityMap<1> sparsity_map = SparsityMap<1>::construct(rects, true, true);
  sparsity_map.add_reference();
  sparsity_map.remove_references();
  auto *impl = sparsity_map.impl();
  EXPECT_NE(impl, nullptr);
  EXPECT_FALSE(impl->is_valid());
}

TEST_F(TransferIteratorTest, AddRemoveZeroReference)
{
  SparsityMap<1> sparsity_map = SparsityMap<1>::construct(rects, true, true);
  sparsity_map.add_reference();
  sparsity_map.remove_references(0);
  auto *impl = sparsity_map.impl();
  EXPECT_NE(impl, nullptr);
  EXPECT_TRUE(impl->is_valid());
}

TEST_F(TransferIteratorTest, DoubleAddRemoveReference)
{
  SparsityMap<1> sparsity_map = SparsityMap<1>::construct(rects, true, true);
  sparsity_map.add_reference();
  sparsity_map.add_reference();
  sparsity_map.remove_references();
  auto *impl = sparsity_map.impl();
  EXPECT_NE(impl, nullptr);
  EXPECT_TRUE(impl->is_valid());
}

TEST_F(TransferIteratorTest, DoubleAddRemoveTwoReference)
{
  SparsityMap<1> sparsity_map = SparsityMap<1>::construct(rects, true, true);
  sparsity_map.add_reference();
  sparsity_map.add_reference();
  sparsity_map.remove_references(2);
  auto *impl = sparsity_map.impl();
  EXPECT_NE(impl, nullptr);
  EXPECT_FALSE(impl->is_valid());
}

TEST_F(TransferIteratorTest, DestroyWithReference)
{
  SparsityMap<1> sparsity_map = SparsityMap<1>::construct(rects, true, true);
  sparsity_map.add_reference();
  sparsity_map.destroy();
  auto *impl = sparsity_map.impl();
  EXPECT_NE(impl, nullptr);
  EXPECT_FALSE(impl->is_valid());
}

TEST_F(TransferIteratorTest, DestroyWithDoubleReference)
{
  SparsityMap<1> sparsity_map = SparsityMap<1>::construct(rects, true, true);
  sparsity_map.add_reference();
  sparsity_map.add_reference();
  sparsity_map.destroy();
  auto *impl = sparsity_map.impl();
  EXPECT_NE(impl, nullptr);
  EXPECT_TRUE(impl->is_valid());
}
