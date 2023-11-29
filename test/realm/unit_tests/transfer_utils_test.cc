#include "realm/transfer/transfer_utils.h"
#include <tuple>
#include <gtest/gtest.h>

using namespace Realm;

class TrasferUtilsTestsWithParams
  : public ::testing::TestWithParam<std::tuple<int, int>> {};

TEST_P(TrasferUtilsTestsWithParams, EmptyDomainAndBounds)
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

TEST_P(TrasferUtilsTestsWithParams, EmptyDomain)
{
  Rect<2> bounds = Rect<2>(Point<2>(0, 0), Point<2>(10, 10));
  Rect<1> subrect, domain = Rect<1>(Point<1>(1), Point<1>(0));
  Point<1> next_start(0);
  std::vector<int> dim_order{0};
  EXPECT_TRUE(
      next_subrect(domain, Point<1>(1), domain, dim_order.data(), subrect, next_start));
  EXPECT_EQ(subrect.lo, domain.lo);
  EXPECT_EQ(subrect.hi, domain.hi);
  EXPECT_EQ(next_start, Point<1>(0));
}

TEST_P(TrasferUtilsTestsWithParams, BoundsContainFullSubrect)
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

TEST_P(TrasferUtilsTestsWithParams, StartNotFullSpanDimension)
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

TEST_P(TrasferUtilsTestsWithParams, HigherDomainBounds)
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

// TODO(apryakhin@):
// 1. base and edge cases with dim_order

INSTANTIATE_TEST_CASE_P(TrasferUtilsTest, TrasferUtilsTestsWithParams,
                        ::testing::Values(std::make_tuple(8, 1024),
                                          std::make_tuple(8, 2048)));
