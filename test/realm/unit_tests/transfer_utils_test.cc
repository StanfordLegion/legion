#include "realm/transfer/transfer_utils.h"
#include <tuple>
#include <variant>
#include <vector>
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

template <int N>
struct ComputeTargetSubrectTestCase {
  std::vector<int> dim_order;
  Rect<N> bounds;
  Rect<N> cur_rect;
  Point<N> cur_point;
  std::vector<Rect<N>> expected_rects;
};

// Define aliases for specific dimensions
using ComputeTargetSubrectTestCase1D = ComputeTargetSubrectTestCase<1>;
using ComputeTargetSubrectTestCase2D = ComputeTargetSubrectTestCase<2>;
using ComputeTargetSubrectTestCase3D = ComputeTargetSubrectTestCase<3>;

// Unified test case variant
using ComputeTargetSubrectTestVariant =
    std::variant<ComputeTargetSubrectTestCase1D, ComputeTargetSubrectTestCase2D,
                 ComputeTargetSubrectTestCase3D>;

class ComputeTargetSubrectTest
  : public ::testing::TestWithParam<ComputeTargetSubrectTestVariant> {
protected:
  void TearDown() override {}

  template <int N>
  void RunTest(const ComputeTargetSubrectTestCase<N> &test_case)
  {
    std::vector<Rect<N>> rects;
    Point<N> cur_point = test_case.cur_point;

    bool not_done = false;
    do {
      Rect<N> next_subrect;
      not_done = compute_target_subrect(test_case.bounds, test_case.cur_rect, cur_point,
                                        next_subrect, test_case.dim_order.data());
      rects.push_back(next_subrect);
    } while(not_done);

    ASSERT_EQ(test_case.expected_rects.size(), rects.size());
    for(size_t i = 0; i < test_case.expected_rects.size(); i++) {
      EXPECT_EQ(test_case.expected_rects[i].lo, rects[i].lo);
      EXPECT_EQ(test_case.expected_rects[i].hi, rects[i].hi);
    }
  }
};

TEST_P(ComputeTargetSubrectTest, NextTargetSubrectEmpty)
{
  std::visit([this](auto &&test_case) { RunTest(test_case); }, GetParam());
}

// 1D test cases
const static ComputeTargetSubrectTestCase1D kComputeSubrectTestCases1D[] = {
    // Simple full span in 1D
    {{0},
     Rect<1>(Point<1>(0), Point<1>(10)),
     Rect<1>(Point<1>(0), Point<1>(10)),
     Point<1>(0),
     {Rect<1>(Point<1>(0), Point<1>(10))}},
    // Partial span in 1D
    {{0},
     Rect<1>(Point<1>(0), Point<1>(10)),
     Rect<1>(Point<1>(3), Point<1>(7)),
     Point<1>(3),
     {Rect<1>(Point<1>(3), Point<1>(7))}},
    // Out of bounds starting point in 1D
    /*{{0},
     Rect<1>(Point<1>(0), Point<1>(10)),
     Rect<1>(Point<1>(11), Point<1>(15)),
     Point<1>(11),
     {}},*/
};

// 2D test cases
const static ComputeTargetSubrectTestCase2D kComputeSubrectTestCases2D[] = {
    // Full cover in 2D
    {{0, 1},
     Rect<2>(Point<2>(0, 0), Point<2>(10, 10)),
     Rect<2>(Point<2>(0, 0), Point<2>(10, 10)),
     Point<2>(0, 0),
     {Rect<2>(Point<2>(0, 0), Point<2>(10, 10))}},
    // Partial span in 2D
    {{0, 1},
     Rect<2>(Point<2>(0, 0), Point<2>(10, 10)),
     Rect<2>(Point<2>(0, 0), Point<2>(10, 5)),
     Point<2>(0, 0),
     {Rect<2>(Point<2>(0, 0), Point<2>(10, 5))}},
    // Middle start point in 2D
    {{0, 1},
     Rect<2>(Point<2>(0, 0), Point<2>(10, 10)),
     Rect<2>(Point<2>(0, 0), Point<2>(10, 5)),
     Point<2>(4, 4),
     {Rect<2>(Point<2>(4, 4), Point<2>(10, 4)),
      Rect<2>(Point<2>(0, 5), Point<2>(10, 5))}},
    // Out of bounds in 2D
    /*{{0, 1},
     Rect<2>(Point<2>(0, 0), Point<2>(10, 10)),
     Rect<2>(Point<2>(11, 11), Point<2>(12, 12)),
     Point<2>(11, 11),
     {}},*/
};

// 3D test cases
const static ComputeTargetSubrectTestCase3D kComputeSubrectTestCases3D[] = {
    // Full cover in 3D
    {{0, 1, 2},
     Rect<3>(Point<3>(0, 0, 0), Point<3>(10, 10, 10)),
     Rect<3>(Point<3>(0, 0, 0), Point<3>(10, 10, 10)),
     Point<3>(0, 0, 0),
     {Rect<3>(Point<3>(0, 0, 0), Point<3>(10, 10, 10))}},
    // Partial span in 3D
    {{0, 1, 2},
     Rect<3>(Point<3>(0, 0, 0), Point<3>(10, 10, 10)),
     Rect<3>(Point<3>(0, 0, 0), Point<3>(10, 5, 5)),
     Point<3>(0, 0, 0),
     {Rect<3>(Point<3>(0, 0, 0), Point<3>(10, 5, 5))}},
    // Middle start in 3D
    {{0, 1, 2},
     Rect<3>(Point<3>(0, 0, 0), Point<3>(10, 10, 10)),
     Rect<3>(Point<3>(0, 0, 0), Point<3>(10, 5, 5)),
     Point<3>(4, 4, 4),
     {Rect<3>(Point<3>(4, 4, 4), Point<3>(10, 4, 4)),
      Rect<3>(Point<3>(0, 5, 4), Point<3>(10, 5, 4)),
      Rect<3>(Point<3>(0, 0, 5), Point<3>(10, 5, 5))}},
    // Out of bounds in 3D
    /*{{0, 1, 2},
     Rect<3>(Point<3>(0, 0, 0), Point<3>(10, 10, 10)),
     Rect<3>(Point<3>(11, 11, 11), Point<3>(12, 12, 12)),
     Point<3>(11, 11, 11),
     {}},*/
};

// Concatenate all test cases into one vector
std::vector<ComputeTargetSubrectTestVariant> allTestCases()
{
  std::vector<ComputeTargetSubrectTestVariant> cases;
  cases.insert(cases.end(), std::begin(kComputeSubrectTestCases1D),
               std::end(kComputeSubrectTestCases1D));
  cases.insert(cases.end(), std::begin(kComputeSubrectTestCases2D),
               std::end(kComputeSubrectTestCases2D));
  cases.insert(cases.end(), std::begin(kComputeSubrectTestCases3D),
               std::end(kComputeSubrectTestCases3D));
  return cases;
}

INSTANTIATE_TEST_SUITE_P(TestAllDimensions, ComputeTargetSubrectTest,
                         testing::ValuesIn(allTestCases()));
