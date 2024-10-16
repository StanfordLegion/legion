#include "realm/transfer/transfer_utils.h"
#include <tuple>
#include <gtest/gtest.h>

using namespace Realm;

TEST(TransferUtilsTest, EmptyDomainAndBounds)
{
  Rect<1> subrect, domain = Rect<1>(Point<1>(1), Point<1>(0));
  Point<1> next_start(0);
  std::vector<int> dim_order{0};
  EXPECT_TRUE(
      next_subrect(domain, Point<1>(1), domain, dim_order.data(), subrect, next_start));
  EXPECT_EQ(subrect.lo, domain.lo);
  EXPECT_EQ(subrect.hi, domain.hi);
  EXPECT_EQ(next_start, Point<1>(0));
}

TEST(TransferUtilsTest, EmptyDomain)
{
  Rect<1> subrect, domain = Rect<1>(Point<1>(1), Point<1>(0));
  Point<1> next_start(0);
  std::vector<int> dim_order{0};
  EXPECT_TRUE(
      next_subrect(domain, Point<1>(1), domain, dim_order.data(), subrect, next_start));
  EXPECT_EQ(subrect.lo, domain.lo);
  EXPECT_EQ(subrect.hi, domain.hi);
  EXPECT_EQ(next_start, Point<1>(0));
}

TEST(TransferUtilsTest, BoundsContainFullSubrect)
{
  Rect<1> subrect, domain = Rect<1>(Point<1>(0), Point<1>(10));
  Point<1> next_start(0);
  std::vector<int> dim_order{0};
  EXPECT_TRUE(
      next_subrect(domain, Point<1>(0), domain, dim_order.data(), subrect, next_start));
  EXPECT_EQ(subrect.lo, domain.lo);
  EXPECT_EQ(subrect.hi, domain.hi);
  EXPECT_EQ(next_start, Point<1>(0));
}

TEST(TransferUtilsTest, StartNotFullSpanDimension)
{
  Rect<2> bounds = Rect<2>(Point<2>(0, 0), Point<2>(10, 10));
  Rect<2> subrect, domain = Rect<2>(Point<2>(0, 0), Point<2>(8, 8));
  Point<2> next_start(0, 0);
  std::vector<int> dim_order{0, 1};

  EXPECT_FALSE(next_subrect(domain, Point<2>(1, 0), bounds, dim_order.data(), subrect,
                            next_start));
  EXPECT_EQ(subrect.lo, Point<2>(1, 0));
  EXPECT_EQ(subrect.hi, Point<2>(8, 0));
  EXPECT_EQ(next_start, Point<2>(0, 1));

  EXPECT_TRUE(
      next_subrect(domain, next_start, bounds, dim_order.data(), subrect, next_start));
  EXPECT_EQ(subrect.lo, Point<2>(0, 1));
  EXPECT_EQ(subrect.hi, Point<2>(8, 8));
  EXPECT_EQ(next_start, Point<2>(0, 0));
}

TEST(TransferUtilsTest, HigherDomainBounds)
{
  Rect<2> bounds = Rect<2>(Point<2>(0, 0), Point<2>(10, 10));
  Rect<2> subrect, domain = Rect<2>(Point<2>(1, 1), Point<2>(11, 11));
  Point<2> next_start(0, 0);
  std::vector<int> dim_order{0, 1};

  for(int next_y = 1; next_y < 10; next_y++) {
    EXPECT_FALSE(next_subrect(domain, Point<2>(1, next_y), bounds, dim_order.data(),
                              subrect, next_start));
    EXPECT_EQ(subrect.lo, Point<2>(1, next_y));
    EXPECT_EQ(subrect.hi, Point<2>(10, next_y));
    EXPECT_EQ(next_start, Point<2>(11, next_y));
  }
}

// TODO(apryakhin@): Add inverted dim order
TEST(TransferUtilsTest, NextTargetSubrectEmpty)
{
  Rect<2> bounds = Rect<2>(Point<2>(1, 1), Point<2>(0, 0));
  Rect<2> cur_rect = Rect<2>(Point<2>(1, 1), Point<2>(0, 0));
  Point<2> cur_point = Point<2>::ZEROES();
  int dim_order[2];
  for(int i = 0; i < 2; i++)
    dim_order[i] = i;

  Rect<2> next_subrect;
  EXPECT_FALSE(
      compute_target_subrect(bounds, cur_rect, cur_point, next_subrect, dim_order));
  EXPECT_EQ(next_subrect.lo, Point<2>(0, 0));
  EXPECT_EQ(next_subrect.hi, Point<2>(0, 0));
}

TEST(TransferUtilsTest, NextTargetSubrectFullCover2D)
{
  Rect<2> bounds = Rect<2>(Point<2>(0, 0), Point<2>(10, 10));
  Rect<2> cur_rect = Rect<2>(Point<2>(0, 0), Point<2>(10, 5));
  Point<2> cur_point = Point<2>::ZEROES();
  int dim_order[2];
  for(int i = 0; i < 2; i++)
    dim_order[i] = i;

  Rect<2> next_subrect;
  EXPECT_FALSE(
      compute_target_subrect(bounds, cur_rect, cur_point, next_subrect, dim_order));
  EXPECT_EQ(next_subrect.lo, Point<2>(0, 0));
  EXPECT_EQ(next_subrect.hi, Point<2>(10, 5));
}

TEST(TransferUtilsTest, NextTargetSubrectMiddleStart2D)
{
  Rect<2> bounds = Rect<2>(Point<2>(0, 0), Point<2>(10, 10));
  Rect<2> cur_rect = Rect<2>(Point<2>(0, 0), Point<2>(10, 5));
  Point<2> cur_point = Point<2>(4, 4);
  int dim_order[2];
  for(int i = 0; i < 2; i++)
    dim_order[i] = i;

  Rect<2> next_subrect;
  EXPECT_TRUE(
      compute_target_subrect(bounds, cur_rect, cur_point, next_subrect, dim_order));
  EXPECT_EQ(next_subrect.lo, Point<2>(4, 4));
  EXPECT_EQ(next_subrect.hi, Point<2>(10, 4));
  EXPECT_EQ(cur_point, Point<2>(0, 5));

  EXPECT_FALSE(
      compute_target_subrect(bounds, cur_rect, cur_point, next_subrect, dim_order));
  EXPECT_EQ(next_subrect.lo, Point<2>(0, 5));
  EXPECT_EQ(next_subrect.hi, Point<2>(10, 5));
}

TEST(TransferUtilsTest, NextTargetSubrectStopShort2D)
{
  Rect<2> bounds = Rect<2>(Point<2>(0, 0), Point<2>(10, 10));
  Rect<2> cur_rect = Rect<2>(Point<2>(0, 0), Point<2>(12, 5));
  Point<2> cur_point = Point<2>(0, 0);
  int dim_order[2];
  for(int i = 0; i < 2; i++)
    dim_order[i] = i;

  Rect<2> next_subrect;
  EXPECT_TRUE(
      compute_target_subrect(bounds, cur_rect, cur_point, next_subrect, dim_order));
  EXPECT_EQ(next_subrect.lo, Point<2>(0, 0));
  EXPECT_EQ(next_subrect.hi, Point<2>(10, 0));
  EXPECT_EQ(cur_point, Point<2>(11, 0));
}
