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

// TODO(apryakhin@): Extend to ND cases
struct ComputeTargetSubrectTestCase {
  std::vector<int> dim_order;
  Rect<2> bounds;
  Rect<2> cur_rect;
  Point<2> cur_point;
  std::vector<Rect<2>> expected_rects;
};

class ComputeTargetSubrectTest
  : public ::testing::TestWithParam<ComputeTargetSubrectTestCase> {
  void TearDown() {}
};

TEST_P(ComputeTargetSubrectTest, NextTargetSubrectEmpty)
{
  auto test_case = GetParam();

  std::vector<Rect<2>> rects;
  Point<2> cur_point = test_case.cur_point;

  bool not_done = false;
  do {
    Rect<2> next_subrect;
    not_done = compute_target_subrect(test_case.bounds, test_case.cur_rect, cur_point,
                                      next_subrect, test_case.dim_order.data());
    rects.push_back(next_subrect);

  } while(not_done);

  EXPECT_EQ(test_case.expected_rects.size(), rects.size());
  for(size_t i = 0; i < test_case.expected_rects.size(); i++) {
    EXPECT_EQ(test_case.expected_rects[i].lo, rects[i].lo);
    EXPECT_EQ(test_case.expected_rects[i].hi, rects[i].hi);
  }
}

const static ComputeTargetSubrectTestCase kComputeSubrectTestCases[] = {
    // Case 1 : next target subrect is empty
    ComputeTargetSubrectTestCase{
        .dim_order = std::vector<int>{0, 1},
        .bounds = Rect<2>(Point<2>(1, 1), Point<2>(0, 0)),
        .cur_rect = Rect<2>(Point<2>(1, 1), Point<2>(0, 0)),
        .cur_point = Point<2>::ZEROES(),
        .expected_rects = {Rect<2>(Point<2>(0, 0), Point<2>(0, 0))}},

    // Case 2 : next target full cover
    ComputeTargetSubrectTestCase{
        .dim_order = std::vector<int>{0, 1},
        .bounds = Rect<2>(Point<2>(0, 0), Point<2>(10, 10)),
        .cur_rect = Rect<2>(Point<2>(0, 0), Point<2>(10, 5)),
        .cur_point = Point<2>::ZEROES(),
        .expected_rects = {Rect<2>(Point<2>(0, 0), Point<2>(10, 5))}},

    // Case 3 : middle start
    ComputeTargetSubrectTestCase{
        .dim_order = std::vector<int>{0, 1},
        .bounds = Rect<2>(Point<2>(0, 0), Point<2>(10, 10)),
        .cur_rect = Rect<2>(Point<2>(0, 0), Point<2>(10, 5)),
        .cur_point = Point<2>(4, 4),
        .expected_rects = {Rect<2>(Point<2>(4, 4), Point<2>(10, 4)),
                           Rect<2>(Point<2>(0, 5), Point<2>(10, 5))}},

    // Case 4 : stopping short
    /*ComputeTargetSubrectTestCase{
        .dim_order = std::vector<int>{0, 1},
        .bounds = Rect<2>(Point<2>(0, 0), Point<2>(10, 10)),
        .cur_rect = Rect<2>(Point<2>(11, 11), Point<2>(12, 12)),
        .cur_point = Point<2>(0, 0),
        .expected_rects = {Rect<2>(Point<2>(0, 0), Point<2>(10, 0))}},*/
};

INSTANTIATE_TEST_SUITE_P(Test, ComputeTargetSubrectTest,
                         testing::ValuesIn(kComputeSubrectTestCases));
