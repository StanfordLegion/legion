#include "realm.h"
#include "realm/transfer/transfer.h"
#include <tuple>
#include <gtest/gtest.h>

using namespace Realm;

class IndexSpaceIteratorTest : public ::testing::Test {
protected:
  static void SetUpTestSuite()
  {
    runtime_ = new Runtime();
    runtime_->init(0, 0);
  }

  static void TearDownTestSuite()
  {
    runtime_->shutdown();
    runtime_->wait_for_shutdown();
    delete runtime_;
  }

  static Runtime *runtime_;
};

Runtime *IndexSpaceIteratorTest::runtime_ = nullptr;

TEST_F(IndexSpaceIteratorTest, CreateWithEmptyIndexSpace)
{
  IndexSpace<1> index_space(Rect<1>(Point<1>(0), Point<1>(-1)));
  IndexSpaceIterator<1> it(index_space);
  EXPECT_FALSE(it.valid);
  EXPECT_TRUE(it.rect.empty());
}

TEST_F(IndexSpaceIteratorTest, StepDense)
{
  IndexSpace<1> index_space(Rect<1>(Point<1>(0), Point<1>(4)));
  IndexSpaceIterator<1> it(index_space);
  for(int i = 0; i < 2; i++) {
    EXPECT_TRUE(it.valid);
    EXPECT_EQ(it.rect.lo, 0);
    EXPECT_EQ(it.rect.hi, 4);
    EXPECT_FALSE(it.step());
    EXPECT_FALSE(it.valid);
    it.reset(index_space);
  }
}

TEST_F(IndexSpaceIteratorTest, CreateEmptyRestrictions)
{
  IndexSpace<1> index_space(Rect<1>(Point<1>(0), Point<1>(16)));
  IndexSpaceIterator<1> it(index_space, Rect<1>(Point<1>(0), Point<1>(-1)));
  EXPECT_FALSE(it.valid);
  EXPECT_EQ(it.rect.lo, 0);
  EXPECT_EQ(it.rect.hi, 16);
}

TEST_F(IndexSpaceIteratorTest, CreateDenseDisjointRestrictions)
{
  IndexSpace<1> index_space(Rect<1>(Point<1>(0), Point<1>(16)));
  IndexSpaceIterator<1> it(index_space, Rect<1>(Point<1>(17), Point<1>(18)));
  EXPECT_FALSE(it.valid);
  EXPECT_EQ(it.rect.lo, 0);
  EXPECT_EQ(it.rect.hi, 16);
}

TEST_F(IndexSpaceIteratorTest, StepDenseWithRestrictions)
{
  IndexSpace<1> index_space(Rect<1>(Point<1>(0), Point<1>(16)));
  IndexSpaceIterator<1> it(index_space, Rect<1>(Point<1>(0), Point<1>(4)));
  for(int i = 0; i < 2; i++) {
    EXPECT_TRUE(it.valid);
    EXPECT_EQ(it.rect.lo, 0);
    EXPECT_EQ(it.rect.hi, 4);
    EXPECT_FALSE(it.step());
    EXPECT_FALSE(it.valid);
    it.reset(index_space, Rect<1>(Point<1>(0), Point<1>(4)));
  }
}

TEST_F(IndexSpaceIteratorTest, StepSparse1D)
{
  std::vector<Rect<1>> rects;
  rects.push_back(Rect<1>(Point<1>(0), Point<1>(4)));
  rects.push_back(Rect<1>(Point<1>(8), Point<1>(12)));
  rects.push_back(Rect<1>(Point<1>(16), Point<1>(18)));
  IndexSpace<1> index_space(rects);
  IndexSpaceIterator<1> it(index_space);
  for(int i = 0; i < 2; i++) {
    EXPECT_TRUE(it.valid);
    EXPECT_EQ(it.rect.lo, rects[0].lo);
    EXPECT_EQ(it.rect.hi, rects[0].hi);
    EXPECT_TRUE(it.step());
    EXPECT_EQ(it.rect.lo, rects[1].lo);
    EXPECT_EQ(it.rect.hi, rects[1].hi);
    EXPECT_TRUE(it.step());
    EXPECT_EQ(it.rect.lo, rects[2].lo);
    EXPECT_EQ(it.rect.hi, rects[2].hi);
    EXPECT_FALSE(it.step());
    EXPECT_FALSE(it.valid);
    it.reset(index_space);
  }
}

TEST_F(IndexSpaceIteratorTest, StepSparse2D)
{
  std::vector<Rect<2>> rects;
  rects.push_back(Rect<2>(Point<2>(0, 0), Point<2>(4, 0)));
  rects.push_back(Rect<2>(Point<2>(0, 2), Point<2>(4, 2)));
  rects.push_back(Rect<2>(Point<2>(0, 4), Point<2>(4, 4)));
  IndexSpace<2> index_space(rects);
  IndexSpaceIterator<2> it(index_space);
  for(int i = 0; i < 2; i++) {
    EXPECT_TRUE(it.valid);
    EXPECT_EQ(it.rect.lo, rects[0].lo);
    EXPECT_EQ(it.rect.hi, rects[0].hi);
    EXPECT_TRUE(it.step());
    EXPECT_EQ(it.rect.lo, rects[1].lo);
    EXPECT_EQ(it.rect.hi, rects[1].hi);
    EXPECT_TRUE(it.step());
    EXPECT_EQ(it.rect.lo, rects[2].lo);
    EXPECT_EQ(it.rect.hi, rects[2].hi);
    EXPECT_FALSE(it.step());
    EXPECT_FALSE(it.valid);
    it.reset(index_space);
  }
}
